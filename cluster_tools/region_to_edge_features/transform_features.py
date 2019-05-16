#! /bin/python

import os
import sys
import json
import pickle

import numpy as np
import luigi

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Summarizes features from regions and edges to create edge features
#

# TODO implement graph extraction with ignore label 0
class R2EFeaturesBase(luigi.Task):
    """ R2EFeaturesBase base class
    """

    task_name = 'r2e_features'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input volumes and graph
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    region_feature_paths = luigi.ListParameter()
    region_feature_keys = luigi.ListParameter()
    edge_feature_paths = luigi.ListParameter(default=None)
    edge_feature_keys = luigi.ListParameter(default=None)
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        return config

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # NOTE we have to turn the luigi dict parameters into normal python dicts
        # in order to json serialize them
        config.update({'graph_path': self.graph_path,
                       'graph_key': self.graph_key,
                       'region_feature_paths': self.region_feature_paths,
                       'region_feature_keys': self.region_feature_keys,
                       'edge_feature_paths': self.edge_feature_paths,
                       'edge_feature_keys': self.edge_feature_keys,
                       'output_path': self.output_path,
                       'output_key': self.output_key})

        # prime and run the jobs
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class R2EFeaturesLocal(R2EFeaturesBase, LocalTask):
    """ R2EFeatures on local machine
    """
    pass


class R2EFeaturesSlurm(R2EFeaturesBase, SlurmTask):
    """ R2EFeatures on slurm cluster
    """
    pass


class R2EFeaturesLSF(R2EFeaturesBase, LSFTask):
    """ R2EFeatures on lsf cluster
    """
    pass


#
# Implementation
#


def r2f_features(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    graph_path = config['graph_path']
    graph_key = config['graph_key']
    region_feature_paths = config['region_feature_paths']
    region_feature_keys = config['region_feature_keys']
    edge_feature_paths = config['edge_feature_paths']
    edge_feature_keys = config['edge_feature_keys']
    output_path = config['output_path']
    output_key = config['output_key']

    # load the uv ids and check
    with vu.file_reader(graph_path, 'r') as f:
        uv_ids = f[graph_key]['edges'][:].astype('int32')
        node_ids = f[graph_key]['nodes'][:]

    num_nodes = len(node_ids)
    num_edges = uv_ids.shape[0]
    
    fu.log("reading edge features from %s%s" % (edge_feature_paths,edge_feature_keys))

    
    num_total_features = 0

    feature_colnames = []

    count = None
    edge_features_list = []
    if len(edge_feature_paths)>0:
        num_total_features = 1
        for ef_set_i in range(len(edge_feature_paths)):
            edge_feature_path = edge_feature_paths[ef_set_i]
            edge_feature_key = edge_feature_keys[ef_set_i]
            with vu.file_reader(edge_feature_path, 'r') as f:
                edge_features = f[edge_feature_key][:]
                edge_features_list.append(edge_features)
                num_edge_features = edge_features.shape[1]-1
                num_total_features += num_edge_features
            feature_colnames = feature_colnames + (['EF'] * (num_edge_features))
            count = edge_features[:,-1]

    region_features = []
    
    for rf_set_i in range(len(region_feature_paths)):
        region_feature_path = region_feature_paths[rf_set_i]
        region_feature_key = region_feature_keys[rf_set_i]
        fu.log("reading edge features from %s%s" % (region_feature_path,region_feature_key))
        with vu.file_reader(region_feature_path, 'r') as f:
            ds_in = f[region_feature_key]
            num_region_features = ds_in.attrs['num_features']
            input_feature_list = ds_in.attrs['feature_indices']
            feature_colnames = feature_colnames + (input_feature_list * 3)
            region_features_vals = ds_in[:].reshape(-1,num_region_features)
            num_nodes = region_features_vals.shape[0]
            
            lu = uv_ids[:, 0]
            lv = uv_ids[:, 1]
            
            rf_u = region_features_vals[lu]
            rf_v = region_features_vals[lv]
            
            region_features.append(np.minimum(rf_u, rf_v))
            region_features.append(np.maximum(rf_u, rf_v))
            region_features.append(np.abs(rf_u-rf_v))
            
            num_total_features += num_region_features * 3
            
    if len(edge_feature_paths)>0:
        feature_colnames.append('EF')
    fu.log("writing output to %s:%s" % (output_path, output_key))
    # require the output dataset
    with vu.file_reader(output_path) as f:
        ds = f.require_dataset(output_key, dtype='float32', shape=(num_edges, num_total_features), compression='gzip')

        total_i=0
        if len(edge_feature_paths)>0:
            for edge_features in edge_features_list:
                # Edge features
                # Note that edge size must remain last feature, so we remove it for now and add it later
                num_edge_features = edge_features.shape[1]-1
                print("Adding edge features at ", total_i, " namely ", num_edge_features)
                ds[:,total_i:(total_i + num_edge_features)] = edge_features[:,:-1]
                total_i += num_edge_features
            
        print("Region features start at ", total_i)
        # Region features
        for rf in region_features:
            num_region_features = rf.shape[1]
            print("Adding region features at ", total_i, " namely ", num_region_features)
            ds[:,total_i:(total_i+num_region_features)] = rf
            total_i+=num_region_features


        print("Now at ", total_i," from ", ds.shape[1])
        if not count is None:
            assert total_i == (ds.shape[1]-1)
            # Now write edge size
            ds[:,-1] = count
        else:
            assert total_i == (ds.shape[1])

        ds.attrs['feature_colnames'] = feature_colnames

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    r2f_features(job_id, path)
