#! /bin/python

import os
import sys
import json

import numpy as np
import luigi
import nifty

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
import cluster_tools.utils.segmentation_utils as su
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask

#
# Multicut Tasks
#


class SolveGlobalBase(luigi.Task):
    """ SolveGlobal base class
    """

    task_name = 'solve_global'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input volumes and graph
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    scale = luigi.IntParameter()
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def clean_up_for_retry(self, block_list):
        # TODO does this work with the mixin pattern?
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'agglomerator': 'kernighan-lin'})
        return config

    def run(self):
        # get the global config and init configs
        self.make_dirs()
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'output_path': self.output_path, 'output_key': self.output_key,
                       'scale': self.scale, 'input_path': self.input_path})

        # prime and run the job
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class SolveGlobalLocal(SolveGlobalBase, LocalTask):
    """ SolveGlobal on local machine
    """
    pass


class SolveGlobalSlurm(SolveGlobalBase, SlurmTask):
    """ SolveGlobal on slurm cluster
    """
    pass


class SolveGlobalLSF(SolveGlobalBase, LSFTask):
    """ SolveGlobal on lsf cluster
    """
    pass


#
# Implementation
#


def solve_global(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)
    input_path = config['input_path']
    output_path = config['output_path']
    output_key = config['output_key']
    scale = config['scale']
    n_threads = config['threads_per_job']
    agglomerator_key = config['agglomerator']

    agglomerator = su.key_to_agglomerator(agglomerator_key)

    # TODO this should come from input variable
    with vu.file_reader(input_path) as f:
        group = f['s%i' % scale]
        n_nodes = group.attrs['numberOfNodes']

        ds = group['edges']
        ds.n_threads = n_threads
        uv_ids = ds[:]
        n_edges = len(uv_ids)

        ds = group['node_labeling']
        ds.n_threads = n_threads
        initial_node_labeling = ds[:]

        ds = group['costs']
        ds.n_threads = n_threads
        costs = ds[:]
        assert len(costs) == n_edges, "%i, %i" (len(costs), n_edges)

    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(uv_ids)
    fu.log("start agglomeration")
    node_labeling = agglomerator(graph, costs)
    # NOTE we don't have a zeros label, so for correct 1d indexing we need to insert it
    node_labeling = np.concatenate((np.zeros(1, dtype=node_labeling.dtype),
                                    node_labeling))

    # get the labeling of initial nodes
    # should this ever become a bottleneck, we can parallelize this in nifty
    # but for now this would really be premature optimization
    new_initial_node_labeling = node_labeling[initial_node_labeling]

    node_shape = (len(new_initial_node_labeling),)
    chunks = (min(len(new_initial_node_labeling), 524288),)
    with vu.file_reader(output_path) as f:
        ds = f.require_dataset(output_key, dtype='uint64',
                               shape=node_shape,
                               chunks=chunks,
                               compression='gzip')
        ds.n_threads = n_threads
        ds[:] = new_initial_node_labeling
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    solve_global(job_id, path)