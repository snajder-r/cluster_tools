import os
import sys
import json
import unittest
import numpy as np
from shutil import rmtree

import luigi
import z5py

import nifty.distributed as ndist
import nifty.graph.opt.lifted_multicut as nlmc

try:
    from ...lifted_features.sparse_lifted_neighborhood import SparseLiftedNeighborhoodLocal
    from ...node_labels import NodeLabelWorkflow
    from ...graph import GraphWorkflow
except ImportError:
    sys.path.append('../..')
    from ...lifted_features.sparse_lifted_neighborhood import SparseLiftedNeighborhoodLocal
    from ...node_labels import NodeLabelWorkflow
    from ...graph import GraphWorkflow


class TestNHWorkflow(unittest.TestCase):
    input_path = '/g/kreshuk/pape/Work/data/cluster_tools_test_data/test_data_lifted.n5'
    ws_key = 'volumes/watershed'
    labels_key = 'volumes/labels'

    tmp_folder = './tmp'
    config_folder = './tmp/configs'
    block_shape = [50, 256, 256]

    @staticmethod
    def _mkdir(dir_):
        try:
            os.mkdir(dir_)
        except OSError:
            pass

    def setUp(self):
        self._mkdir(self.tmp_folder)
        self._mkdir(self.config_folder)
        global_config = SparseLiftedNeighborhoodLocal.default_global_config()
        global_config['shebang'] = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
        global_config['block_shape'] = self.block_shape
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def compute_nh(self, graph_depth):
        # load the graph
        graph_path = os.path.join(self.tmp_folder, 'graph.n5', 'graph')
        graph = ndist.loadAsUndirectedGraph(graph_path)

        # load the node labels
        node_label_path = os.path.join(self.tmp_folder, 'node_labels.n5')
        node_label_key = 'node_labels'
        node_labels = z5py.File(node_label_path)[node_label_key][:]
        self.assertEqual(len(node_labels), graph.numberOfNodes)

        # run bfs up to depth 4 to get complete lifted nh
        lifted_graph = nlmc.liftedMulticutObjective(graph)
        lifted_graph.insertLiftedEdgesBfs(graph_depth)
        nh = lifted_graph.liftedUvIds()

        # filter by nodes which have a node labeling
        node_ids = np.arange(len(node_labels))
        nodes_with_label = node_ids[node_labels != 0]
        nh_mask = np.in1d(nh, nodes_with_label).reshape(nh.shape)
        nh_mask = nh_mask.all(axis=1)
        nh = nh[nh_mask]
        # need to lexsort - awkward in numpy ...
        nh = nh[np.lexsort(np.rot90(nh))]
        return nh

    def _check_result(self, graph_depth):
        # compute nh in memory
        nh = self.compute_nh(graph_depth)
        # load the nh
        out_path = os.path.join(self.tmp_folder, 'lifted_nh.h5')
        out_key = 'lifted_nh'
        nh_out = z5py.File(out_path)[out_key][:]
        # check that results agree
        self.assertEqual(nh_out.shape, nh.shape)
        self.assertTrue(np.allclose(nh_out, nh))

    def test_lifted_nh_with_labels(self):
        node_label_path = os.path.join(self.tmp_folder, 'node_labels.n5')
        node_label_key = 'node_labels'
        task_labels = NodeLabelWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                                        max_jobs=8, target='local',
                                        ws_path=self.input_path, ws_key=self.ws_key,
                                        input_path=self.input_path, input_key=self.labels_key,
                                        output_path=node_label_path, output_key=node_label_key,
                                        max_overlap=True)

        graph_path = os.path.join(self.tmp_folder, 'graph.n5')
        graph_key = 'graph'
        graph_config = GraphWorkflow.get_config()['initial_sub_graphs']
        graph_config["ignore_label"] = False
        with open(os.path.join(self.config_folder, 'initial_sub_graphs.config'), 'w') as f:
            json.dump(graph_config, f)

        task_graph = GraphWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                                   max_jobs=8, target='local',
                                   input_path=self.input_path, input_key=self.ws_key,
                                   graph_path=graph_path, output_key=graph_key)
        ret = luigi.build([task_labels, task_graph],
                          local_scheduler=True)
        self.assertTrue(ret)

        # TODO try different graph depth and different number of threads !
        graph_depth = 3
        out_path = os.path.join(self.tmp_folder, 'lifted_nh.h5')
        out_key = 'lifted_nh'
        task_nh = SparseLiftedNeighborhoodLocal(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                                                max_jobs=1, dependency=task_graph,
                                                graph_path=graph_path, graph_key=graph_key,
                                                node_label_path=node_label_path, node_label_key=node_label_key,
                                                output_path=out_path, output_key=out_key,
                                                prefix='', nh_graph_depth=graph_depth)
        ret = luigi.build([task_nh], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result(graph_depth)

    def test_lifted_nh(self):
        graph_path = os.path.join(self.tmp_folder, 'graph.n5')
        graph_key = 'graph'
        graph_config = GraphWorkflow.get_config()['initial_sub_graphs']
        graph_config["ignore_label"] = False
        with open(os.path.join(self.config_folder, 'initial_sub_graphs.config'), 'w') as f:
            json.dump(graph_config, f)
        task_graph = GraphWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                                   max_jobs=8, target='local',
                                   input_path=self.input_path, input_key=self.ws_key,
                                   graph_path=graph_path, output_key=graph_key)
        ret = luigi.build([task_graph],
                          local_scheduler=True)
        self.assertTrue(ret)

        graph = ndist.loadAsUndirectedGraph(os.path.join(graph_path, graph_key))
        n_nodes = graph.numberOfNodes
        node_labels = np.ones(n_nodes, dtype='uint64')

        node_label_path = os.path.join(self.tmp_folder, 'node_labels.n5')
        node_label_key = 'node_labels'
        with z5py.File(node_label_path) as f:
            ds = f.require_dataset(node_label_key, shape=node_labels.shape, dtype=node_labels.dtype,
                                   chunks=(1000,), compression='gzip')
            ds[:] = node_labels

        # TODO try different graph depth and different number of threads !
        graph_depth = 3
        out_path = os.path.join(self.tmp_folder, 'lifted_nh.h5')
        out_key = 'lifted_nh'
        task_nh = SparseLiftedNeighborhoodLocal(tmp_folder=self.tmp_folder, config_dir=self.config_folder,
                                                max_jobs=1, dependency=task_graph,
                                                graph_path=graph_path, graph_key=graph_key,
                                                node_label_path=node_label_path, node_label_key=node_label_key,
                                                output_path=out_path, output_key=out_key,
                                                prefix='', nh_graph_depth=graph_depth)
        ret = luigi.build([task_nh], local_scheduler=True)
        self.assertTrue(ret)
        self._check_result(graph_depth)


if __name__ == '__main__':
    unittest.main()
