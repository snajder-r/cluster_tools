#! /bin/python

import os
import sys
import json
import pickle

import numpy as np
import luigi
import skimage

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class GrowRegionsTaskBase(luigi.Task):
    """ GrowregionsTask base class
    """

    task_name = 'grow_regions'
    src_file = os.path.abspath(__file__)

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    de_labels_path = luigi.Parameter()
    de_labels_key = luigi.Parameter()
    boundaries_path = luigi.Parameter()
    boundaries_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()

    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
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
        config.update({'input_path': self.input_path, 'input_key':self.input_key,
                       'de_labels_path': self.de_labels_path, 'de_labels_key':self.de_labels_key,
                       'boundaries_path': self.boundaries_path, 'boundaries_key':self.boundaries_key,
                       'graph_path': self.graph_path, 'graph_key':self.graph_key,
                       'output_path': self.output_path, 'output_key': self.output_key})

        # prime and run the jobs
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class GrowRegionsTaskLocal(GrowRegionsTaskBase, LocalTask):
    """ GrowRegionsTask on local machine
    """
    pass


class GrowRegionsTaskSlurm(GrowRegionsTaskBase, SlurmTask):
    """ GrowRegionsTask on slurm cluster
    """
    pass


class GrowRegionsTaskLSF(GrowRegionsTaskBase, LSFTask):
    """ GrowRegionsTask on lsf cluster
    """
    pass


#
# Implementation
#


def grow_regions(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    input_path = config['input_path']
    input_key = config['input_key']
    de_labels_path = config['de_labels_path']
    de_labels_key = config['de_labels_key']
    boundaries_path = config['boundaries_path']
    boundaries_key = config['boundaries_key']
    graph_path = config['graph_path']
    graph_key = config['graph_key']
    output_path = config['output_path']
    output_key = config['output_key']

    with vu.file_reader(input_path) as f:
        input = f[input_key][:]
    with vu.file_reader(de_labels_path) as f:
        de_labels = f[de_labels_key][:]
    with vu.file_reader(boundaries_path) as f:
        boundaries = f[boundaries_key][:]
    with vu.file_reader(graph_path) as f:
        nodes = f[graph_key]['nodes'][:]
        edges = f[graph_key]['edges'][:]


    maxId = int(input.max())

    edge_labels_u = de_labels[edges[:,0]]  
    edge_labels_v = de_labels[edges[:,1]]

    # Class 0 is fluid cavity, class 2 is background, class 1 is cell
    double_edges = np.logical_or(np.logical_and(edge_labels_u == 0, edge_labels_v == 2), np.logical_and(edge_labels_u == 2, edge_labels_v == 0))

    double_edges = edges[double_edges,:]

    out = input.copy()

    for i in range(double_edges.shape[0]):
        de_node_a = double_edges[i][0]
        de_node_b = double_edges[i][1]

        a = skimage.morphology.dilation(input==de_node_a)
        b = skimage.morphology.dilation(input==de_node_b)
        edge_pixels = np.logical_and(a,b)
        edge_pixels = skimage.morphology.dilation(np.logical_and(a,b))

        seed_values = input[edge_pixels] 

        markers = np.zeros(input.shape)

        for seed_value in np.unique(seed_values):
            if seed_value == de_node_a or seed_value == de_node_b:
                continue
            seed_pos = np.logical_and(input == seed_value, edge_pixels)
            fu.log('Seeding %d at %d positions' % (seed_value, seed_pos.sum()))
            markers[seed_pos] = seed_value

        ws_result = skimage.morphology.watershed(boundaries, markers, mask=edge_pixels)

        out[ws_result > 0] = ws_result[ws_result > 0]

    with vu.file_reader(output_path,'w') as f:
        ds = f.require_dataset(output_key, dtype='uint32', shape=out.shape, compression='gzip')
        ds[:] = out
        ds.attrs['num_fluid_cavities'] = int((de_labels==0).reshape(-1).sum())

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    grow_regions(job_id, path)
