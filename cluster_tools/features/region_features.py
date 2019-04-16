#! /usr/bin/python

import os
import sys
import json

import numpy as np
import luigi
import z5py
import nifty.tools as nt
import vigra

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


# TODO support multi-channel inputs
# TODO for now we only support 'mean', but we should at
# least support all trivially mergeable region features
class RegionFeaturesBase(luigi.Task):
    """ Block edge feature base class
    """

    task_name = 'region_features'
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    labels_path = luigi.Parameter()
    labels_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    feature_list = luigi.ListParameter()
    blockwise = luigi.Parameter(True)
    dependency = luigi.TaskParameter()
    
    
    
    def requires(self):
        return self.dependency

    # TODO specify which features to use here
    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'ignore_label': 0})
        return config

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        if self.blockwise == False:
            with vu.file_reader(self.input_path) as f_in:
                block_shape = f_in[self.input_key].shape
        
        # load the task config
        config = self.get_task_config()

       
        # TODO make the scale at which we extract features accessible
        # update the config with input and output paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'labels_path': self.labels_path, 'labels_key': self.labels_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'block_shape': block_shape, 'feature_list': self.feature_list})
        # TODO support multi-channel
        shape = vu.get_shape(self.input_path, self.input_key)

        # require the temporary output data-set
        f_out = z5py.File(self.output_path)
        
        f_out.require_dataset(self.output_key, shape=shape, compression='gzip',
                              chunks=tuple(block_shape), dtype='float32')

        if self.n_retries == 0:
            # get shape and make block config
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        n_jobs = min(len(block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class RegionFeaturesLocal(RegionFeaturesBase, LocalTask):
    """ RegionFeatures on local machine
    """
    pass


class RegionFeaturesSlurm(RegionFeaturesBase, SlurmTask):
    """ RegionFeatures on slurm cluster
    """
    pass


class RegionFeaturesLSF(RegionFeaturesBase, LSFTask):
    """ RegionFeatures on lsf cluster
    """
    pass


#
# Implementation
#


def _block_features(block_id, blocking,
                    ds_in, ds_labels, ds_out,
                    ignore_label, feature_list):
    fu.log("start processing block %i" % block_id)
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)

    labels = ds_labels[bb]

    # check if we have an ignore label and return
    # if this block is purely ignore label
    if ignore_label is not None:
        if np.sum(labels != ignore_label) == 0:
            fu.log_block_success(block_id)
            return

    # TODO support multichannel
    # get global normalization values
    min_val = 0
    max_val = 255. if ds_in.dtype == np.dtype('uint8') else 1.

    input_ = vu.normalize(ds_in[bb], min_val, max_val)
    # TODO support more features
    # TODO we might want to check for overflows and in general allow vigra to
    # work with uint64s ...
    
    labels = labels.astype('uint32')
    #print(vigra.analysis.supportedFeatures(input_))
    feats = vigra.analysis.extractRegionFeatures(input_, labels, features=feature_list,
                                                 ignoreLabel=ignore_label)
    # make serialization
    num_features = len(feature_list) 
    ids = np.unique(labels)

    # Number of actual feature values, since some features (like Histogram) compute multiple values
    num_feature_vals = 1 # start with 1 for ids
    for i in range(0,num_features):
        feature_vals = feats[feature_list[i]]
        if(len(feature_vals.shape)==2):
            num_feature_vals += feature_vals.shape[1]
        else:
            num_feature_vals+=1
    
    data = np.zeros((len(ids), num_feature_vals), dtype='float32')
    # write the ids
    data[:,0] = ids.astype('float32')

    feature_indices = [''] * num_feature_vals

    # write features
    feature_i = 1
    for i in range(num_features):
        feature_vals = feats[feature_list[i]][ids]
        if len(feature_vals.shape) == 2:
            for j in range(feature_vals.shape[1]):
                data[:,feature_i] = feature_vals[:,j]
                feature_indices[feature_i] = feature_list[i]
                feature_i+=1
        else:
            data[:,feature_i] = feature_vals
            feature_indices[feature_i] = feature_list[i]
            feature_i+=1

    data = data.reshape((-1))
    # z5py cant handle nan, so we need to replace it
    data[np.isnan(data)] = 0 
    chunks = blocking.blockShape
    chunk_id = tuple(b.start // ch for b, ch in zip(bb, chunks))
    ds_out.write_chunk(chunk_id, data, True)
    ds_out.attrs['feature_indices'] = feature_indices
    fu.log_block_success(block_id)


def region_features(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path, 'r') as f:
        config = json.load(f)

    block_list = config['block_list']
    input_path = config['input_path']
    input_key = config['input_key']
    labels_path = config['labels_path']
    labels_key = config['labels_key']
    
    output_path = config['output_path']
    output_key = config['output_key']
    block_shape = config['block_shape']
    ignore_label = config['ignore_label']
    feature_list = config['feature_list']

    with vu.file_reader(input_path) as f_in,\
            vu.file_reader(labels_path) as f_l,\
            vu.file_reader(output_path) as f_out:

        ds_in = f_in[input_key]
        ds_labels = f_l[labels_key]
        ds_out = f_out[output_key]

        shape = ds_out.shape
        
        print('Block shape ', block_shape)
        blocking = nt.blocking([0, 0, 0], shape, block_shape)

        for block_id in block_list:
            _block_features(block_id, blocking,
                            ds_in, ds_labels, ds_out,
                            ignore_label, feature_list)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    region_features(job_id, path)
