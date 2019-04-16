#! /usr/bin/python

import os
import sys
import json

import numpy as np
import luigi
import nifty.tools as nt
import vigra

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class MergeRegionFeaturesBase(luigi.Task):
    """ Merge edge feature base class
    """

    task_name = 'merge_region_features'
    src_file = os.path.abspath(__file__)
    # retry is too complecated for now ...
    allow_retry = False

    # input and output volumes
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    block_features_path = luigi.Parameter()
    block_features_key = luigi.Parameter()
    number_of_labels = luigi.IntParameter()
    dependency = luigi.TaskParameter()
    feature_list = luigi.ListParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()
        chunk_size = min(10000, self.number_of_labels)

        # Number of actual feature values, since some features (like Histogram) compute multiple values
        num_features = len(self.feature_list)
        num_feature_vals = 0
        dummy_feats =  vigra.analysis.extractRegionFeatures(np.zeros((10,10,10),dtype='float32'), np.zeros((10,10,10),dtype='uint32'), features=self.feature_list)
        for i in range(0,num_features):
            feature_vals = dummy_feats[self.feature_list[i]]
            if(len(feature_vals.shape)==2):
                num_feature_vals += feature_vals.shape[1]
            else:
                num_feature_vals+=1
            
        
        # require the output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, dtype='float32', shape=(self.number_of_labels * num_feature_vals,),
                              chunks=(chunk_size,), compression='gzip')



        # update the task config
        config.update({'output_path': self.output_path, 'output_key': self.output_key,
                       'tmp_path': self.block_features_path, 'tmp_key': self.block_features_key,
                       'node_chunk_size': chunk_size, 'feature_list': self.feature_list, 
                       'num_feature_vals':num_feature_vals})

        node_block_list = vu.blocks_in_volume([self.number_of_labels], [chunk_size])

        n_jobs = min(len(node_block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, node_block_list, config, consecutive_blocks=True)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class MergeRegionFeaturesLocal(MergeRegionFeaturesBase, LocalTask):
    """ MergeRegionFeatures on local machine
    """
    pass


class MergeRegionFeaturesSlurm(MergeRegionFeaturesBase, SlurmTask):
    """ MergeRegionFeatures on slurm cluster
    """
    pass


class MergeRegionFeaturesLSF(MergeRegionFeaturesBase, LSFTask):
    """ MergeRegionFeatures on lsf cluster
    """
    pass


#
# Implementation
#

def _extract_and_merge_region_features(blocking, ds_in, ds, node_begin, node_end, feature_list, num_feature_vals):
    fu.log("processing node range %i to %i" % (node_begin, node_end))

    num_features = len(feature_list) 
    num_nodes = node_end - node_begin

    out_features = np.zeros(num_nodes * (num_features), dtype='float32')
    out_count = np.zeros(num_nodes, dtype='float32')

    chunks = ds_in.chunks
    
    for block_id in range(blocking.numberOfBlocks):
        block = blocking.getBlock(block_id)
        chunk_id = tuple(beg // ch for beg, ch in zip(block.begin, chunks))

        # load the data
        data = ds_in.read_chunk(chunk_id)
        if data is None:
            continue

        # TODO support more features
        # extract ids and features
        ids = data[::(num_features+1)].astype('uint64')
        assert 'count' in feature_list
        count = data[(feature_list.index('count')+1)::(num_features+1)]

        # check if any ids overlap with our id range
        overlap_mask = np.logical_and(ids >= node_begin,
                                      ids < node_end)
        if np.sum(overlap_mask) == 0:
            continue


        overlapping_ids = ids[overlap_mask]
        overlapping_ids -= node_begin
        overlapping_count = count[overlap_mask]
        
        

        prev_count = out_count[overlapping_ids]
        tot_count = (prev_count + overlapping_count)

        for i in range(num_features):
            feature_name = feature_list[i]

            prev_feat = out_features[i::(num_features)][overlapping_ids]
            cur_feat = data[(i+1)::(num_features+1)][overlap_mask]

            
            if feature_name == 'mean':
                # calculate cumulative moving average
                out_feats = (overlapping_count * cur_feat + prev_count * prev_feat) / tot_count
            if feature_name == 'minimum':
                out_feats = (np.minimum(prev_feat,cur_feat))
            if feature_name == 'maximum':
                out_feats = (np.maximum(prev_feat,cur_feat))
            if feature_name == 'count':
                out_feats = prev_feat + cur_feat

            out_features[i::(num_features)][overlapping_ids] = out_feats

        out_count[overlapping_ids] += overlapping_count

    ds[(node_begin*num_features):(node_end*num_features)] = out_features


    
def _extract_single_block_region_features(ds_in, ds, node_begin, node_end, feature_list, num_feature_vals):
    fu.log("processing single block")

    # load the data
    data = ds_in.read_chunk((0,0,0))
    
    num_total_nodes = node_end - node_begin
     
    #+1 because ids are an additional column
    print(num_feature_vals)
    ids = data[::(num_feature_vals+1)].astype('uint64')
    print(ids)

    out_features = np.zeros((num_total_nodes*num_feature_vals), dtype='float32')
    
    # only need to remove the id
    for i in range(num_feature_vals):
        out_features[i::(num_feature_vals)][ids] = data[(i+1)::(num_feature_vals+1)]

    ds[:] = out_features
    ds.attrs['feature_indices'] = ds_in.attrs['feature_indices'][1:]

    
def merge_region_features(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path, 'r') as f:
        config = json.load(f)
    output_path = config['output_path']
    output_key = config['output_key']
    tmp_path = config['tmp_path']
    tmp_key = config['tmp_key']
    node_block_list = config['block_list']
    node_chunk_size = config['node_chunk_size']
    feature_list = config['feature_list']
    num_feature_vals = config['num_feature_vals']

    with vu.file_reader(output_path) as f,\
            vu.file_reader(tmp_path) as f_in:

        ds_in = f_in[tmp_key]
        ds = f[output_key]
        ds.attrs['num_features'] = num_feature_vals
        n_nodes = ds.shape[0]

        node_blocking = nt.blocking([0], [n_nodes], [node_chunk_size])
        node_begin = node_blocking.getBlock(node_block_list[0]).begin[0]
        node_end = node_blocking.getBlock(node_block_list[-1]).end[0]

        shape = list(ds_in.shape)
        chunks = list(ds_in.chunks)
        blocking = nt.blocking([0, 0, 0], shape, chunks)

        if blocking.numberOfBlocks == 1:
            _extract_single_block_region_features(ds_in, ds, node_begin, node_end, feature_list, num_feature_vals)
        else:
            _extract_and_merge_region_features(blocking, ds_in, ds, node_begin, node_end, feature_list,num_feature_vals)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    merge_region_features(job_id, path)
