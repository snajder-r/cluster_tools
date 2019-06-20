#! /bin/python

import os
import sys
import json
import pickle

import numpy as np
import luigi
import skimage
import scipy

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class RemoveNoiseObjectsTaskBase(luigi.Task):
    """ RemoveNoiseObjects base class
    """

    task_name = 'remove_noise_object'
    src_file = os.path.abspath(__file__)

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()

    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    
    output_graph_path = luigi.Parameter()
    output_graph_key = luigi.Parameter()
    
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
                       'graph_path': self.graph_path, 'graph_key':self.graph_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'output_graph_path': self.output_graph_path, 'output_graph_key': self.output_graph_key})

        # prime and run the jobs
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class RemoveNoiseObjectsTaskLocal(RemoveNoiseObjectsTaskBase, LocalTask):
    """ RemoveNoiseObjectsTask on local machine
    """
    pass


class RemoveNoiseObjectsTaskSlurm(RemoveNoiseObjectsTaskBase, SlurmTask):
    """ RemoveNoiseObjectsTask on slurm cluster
    """
    pass


class RemoveNoiseObjectsTaskLSF(RemoveNoiseObjectsTaskBase, LSFTask):
    """ RemoveNoiseObjectsTask on lsf cluster
    """
    pass


#
# Implementation
#


def remove_noise_objects(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    input_path = config['input_path']
    input_key = config['input_key']
    graph_path = config['graph_path']
    graph_key = config['graph_key']
    output_path = config['output_path']
    output_key = config['output_key']
    output_graph_path = config['output_graph_path']
    output_graph_key = config['output_graph_key']

    with vu.file_reader(input_path) as f:
        input = f[input_key][:]
    with vu.file_reader(graph_path) as f:
        nodes = f[graph_key]['nodes'][:]
        edges = f[graph_key]['edges'][:]


    # First, remove all components except the largest one
    bg_candidates = [input[0,0,0], input[0,0,-1], input[0,-1,0], input[0,-1,-1], input[-1,0,0], input[-1,0,-1], input[-1,-1,0], input[-1,-1,-1]]
    bg_id = max(bg_candidates, key=bg_candidates.count)
    fu.log("Background id is %d" % bg_id)
    
    adj_m = np.zeros((len(nodes), len(nodes)))
    for edge in edges:
        if edge[0] != bg_id and edge[1] != bg_id:
            adj_m[edge[0], edge[1]] = 1
    
    n_comp, comp = scipy.sparse.csgraph.connected_components(adj_m, directed=False, return_labels=True)   
    fu.log("Found %d connected components" % n_comp)
    
    max_size = 0
    max_comp = 0
    for component in set(comp):
        if component != bg_id:
            compsize = 0
            for label in nodes[comp==component]:
                compsize += (input == label).sum()
            if compsize > max_size:
                max_size = compsize
                max_comp = component

    to_remove = nodes[comp!=max_comp]
    for rc in to_remove:
        fu.log("Cleaning up id %d" % rc)
        if rc != bg_id:
            input[input==rc] = bg_id
            nodes[nodes==rc] = bg_id
            edges[edges==rc] = bg_id
    
    # Remove duplicates from graph
    nodes = np.unique(nodes)
    edge_new = []
    for edge in edges:
        if edge[0] == edge[1]:
            pass
        elif edge[1] < edge[0]:
            edge_new.append([edge[1], edge[0]])
        else:
            edge_new.append([edge[0], edge[1]])
            
    edges = np.array(edge_new, dtype=edge.dtype)
    edges = np.unique(edge_new, axis=0)
        
        

    # Now do the size based merging, as a method to compensate for some leftover oversegmentation
    # Skip this step if we have too few objects (since that means we are early in the development
    # of the embryo, and there are these small polar bodies, which we dont want to merge with the cells)
    if len(nodes) >= 10:
        sizes = []
        for node in nodes:
            sizes.append((input==node).sum())
        size_median = np.median(sizes)
        min_cell_size = size_median * 0.3
        
        sizes = np.array(sizes)
        node_sorted = np.argsort(sizes)
        for i in range(len(node_sorted)):
            node = node_sorted[i]
            if sizes[node] < min_cell_size:
                fu.log('Cell %d is too small '%node)
                neighbor_edges = edges[np.logical_or(edges[:,0]==node, edges[:,1]==node)]
                neighbors = neighbor_edges[neighbor_edges != node].reshape(-1)
                if len(neighbors)>0:
                    smallest_neighbor_i = np.argmin(sizes[neighbors])
                    smallest_neighbor = neighbors[smallest_neighbor_i]
                    input[input==node] = smallest_neighbor
                    sizes[smallest_neighbor] += sizes[node]
                    sizes[node] = 0
                    node_sorted = np.argsort(sizes)
                    edges[edges==node] = smallest_neighbor
                    fu.log('Merging %d with %d'% (node, smallest_neighbor))
    
        
    with vu.file_reader(output_path,'w') as f:
        ds = f.require_dataset(output_key, dtype='uint32', shape=input.shape, compression='gzip')
        ds[:] = input
        
    with vu.file_reader(output_graph_path,'w') as f:
        ds = f.require_dataset(output_graph_key+"/nodes", dtype='uint32', shape=nodes.shape, compression='gzip')
        ds[:] = nodes
        ds = f.require_dataset(output_graph_key+"/edges", dtype='uint32', shape=edges.shape, compression='gzip')
        ds[:] = edges
        
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    remove_noise_objects(job_id, path)
