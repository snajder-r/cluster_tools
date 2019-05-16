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


class ConsequtiveLabelsBase(luigi.Task):
    task_name = 'conseq_labels'
    src_file = os.path.abspath(__file__)
    # retry is too complecated for now ...
    allow_retry = False

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
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
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'output_path': self.output_path, 'output_key': self.output_key
                       })


        # prime and run the jobs
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class ConsequtiveLabelsLocal(ConsequtiveLabelsBase, LocalTask):
    """ ConsequtiveLabels on local machine
    """
    pass


class ConsequtiveLabelsSlurm(ConsequtiveLabelsBase, SlurmTask):
    """ ConsequtiveLabels on slurm cluster
    """
    pass


class ConsequtiveLabelsLSF(ConsequtiveLabelsBase, LSFTask):
    """ ConsequtiveLabels on lsf cluster
    """
    pass


#
# Implementation
#


# TODO parallelize
def conseq_labels(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    print("RUN")
    # get the config
    with open(config_path, 'r') as f:
        config = json.load(f)
    output_path = config['output_path']
    output_key = config['output_key']
    input_path = config['input_path']
    input_key = config['input_key']

    # load the labels
    with vu.file_reader(input_path,'r') as f:
        labels = f[input_key][:]
        unique = np.unique(labels)
        output = np.zeros(labels.shape, dtype=labels.dtype)
        next_l = 0
        for l in sorted(list(unique)):
            output[labels==l] = next_l
            next_l = next_l + 1

        with vu.file_reader(output_path,'w') as fout:
            fout.create_dataset(output_key, data=output, chunks=f[input_key].chunks, compression='gzip')

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    conseq_labels(job_id, path)
