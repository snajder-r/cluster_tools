#! /usr/bin/python

import os
import sys
import argparse
import json

import numpy as np
import luigi
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class MergeRegionFeaturesBase(luigi.Task):

    task_name = 'merge_region_features'
    src_file = os.path.abspath(__file__)
    # retry is too complecated for now ...
    allow_retry = False

    # input and output volumes
    feature_path = luigi.Parameter()
    feature_keys = luigi.ListParameter()
    output_key = luigi.Parameter()
    feature_list = luigi.ListParameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang, _, _, _ = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the task config
        config.update({'feature_path': self.feature_path, 'feature_keys': self.feature_keys,
                       'output_key': self.output_key, 'feature_list': self.feature_list})

        # prime and run the jobs
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class  MergeRegionFeaturesLocal(MergeRegionFeaturesBase, LocalTask):
    """ EdgeLabels on local machine
    """
    pass


class  MergeRegionFeaturesSlurm(MergeRegionFeaturesBase, SlurmTask):
    """ EdgeLabels on slurm cluster
    """
    pass


class  MergeRegionFeaturesLSF(MergeRegionFeaturesBase, LSFTask):
    """ EdgeLabels on lsf cluster
    """
    pass


#
# Implementation
#


# TODO parallelize
def merge(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path, 'r') as f:
        config = json.load(f)
    feature_path = config['feature_path']
    feature_list = config['feature_list']
    output_key = config['output_key']
    feature_keys = config['feature_keys']
    
    count = None
    with vu.file_reader(feature_path, 'a') as f:
        features = list()
        feature_colnames = list()
        for feature_key in feature_keys:
            ds_in = f[feature_key]
            num_region_features = ds_in.attrs['num_features']
            input_feature_list = ds_in.attrs['feature_indices']
            region_features_vals = ds_in[:].reshape(-1,num_region_features)
            feature_colnames = feature_colnames + input_feature_list

            features.append(region_features_vals[:,:]) 

        merged = np.concatenate(features, axis=1)


        assert merged.shape[1] == len(feature_colnames)

        f.require_dataset(output_key, shape=merged.shape, compression='gzip', dtype='float32')
        f[output_key][:] = merged
        f[output_key].attrs['feature_colnames'] = feature_colnames

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    merge(job_id, path)
