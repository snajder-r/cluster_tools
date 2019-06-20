import os
import sys
import json
import unittest
import numpy as np
from shutil import rmtree

import luigi
import z5py
import vigra
import nifty.tools as nt

try:
    from ...features import RegionFeaturesWorkflow
except ImportError:
    sys.path.append('../..')
    from ...features import RegionFeaturesWorkflow


class TestEdgeFeatures(unittest.TestCase):
    # input_path = '/g/kreshuk/pape/Work/data/cluster_tools_test_data/test_data.n5'
    input_path = '/home/pape/Work/data/cluster_tools_test_data/test_data.n5'
    input_key = 'volumes/raw'
    seg_key = 'volumes/watershed'

    tmp_folder = './tmp'
    output_path = './tmp/features.n5'
    output_key = 'features'
    config_folder = './tmp/configs'
    target = 'local'
    block_shape = [10, 256, 256]

    @staticmethod
    def _mkdir(dir_):
        try:
            os.mkdir(dir_)
        except OSError:
            pass

    def setUp(self):
        self._mkdir(self.tmp_folder)
        self._mkdir(self.config_folder)
        global_config = RegionFeaturesWorkflow.get_config()['global']
        # global_config['shebang'] = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python'
        global_config['shebang'] = '#! /home/pape/Work/software/conda/miniconda3/envs/main/bin/python'
        global_config['block_shape'] = self.block_shape
        with open(os.path.join(self.config_folder, 'global.config'), 'w') as f:
            json.dump(global_config, f)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _check_features(self, data, labels, res, ids=None, feat_name='mean'):
        expected = vigra.analysis.extractRegionFeatures(data, labels, features=[feat_name])
        expected = expected[feat_name]

        if ids is not None:
            expected = expected[ids]

        self.assertEqual(res.shape, expected.shape)
        self.assertTrue(np.allclose(res, expected))

    def _check_result(self):
        # load the result
        with z5py.File(self.output_path) as f:
            res = f[self.output_key][:]
        # compute the vigra result
        with z5py.File(self.input_path) as f:
            inp = f[self.input_key][:]
            seg = f[self.seg_key][:].astype('uint32')
        self._check_features(inp, seg, res)

    def _check_subresults(self):
        f = z5py.File(self.input_path)
        dsi = f[self.input_key]
        dsl = f[self.seg_key]
        blocking = nt.blocking([0, 0, 0], dsi.shape, self.block_shape)

        f_feat = z5py.File(os.path.join(self.tmp_folder, 'region_features_tmp.n5'))
        ds_feat = f_feat['block_feats']

        n_blocks = blocking.numberOfBlocks
        for block_id in range(n_blocks):
            # print("Checking block", block_id, "/", n_blocks)
            block = blocking.getBlock(block_id)
            bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
            inp = dsi[bb]
            seg = dsl[bb].astype('uint32')

            # load the sub-result
            chunk_id = tuple(beg // bs for beg, bs in zip(block.begin, self.block_shape))
            res = ds_feat.read_chunk(chunk_id)
            self.assertFalse(res is None)

            # check that ids are correct
            ids = res[::3].astype('uint32')
            expected_ids = np.unique(seg)
            self.assertEqual(ids.shape, expected_ids.shape)
            self.assertTrue(np.allclose(ids, expected_ids))

            # check that mean is correct
            mean = res[2::3]
            self._check_features(inp, seg, mean, ids)

            # check that counts are correct
            counts = res[1::3]
            self._check_features(inp, seg, counts, ids,
                                 feat_name='count')

    def test_region_features(self):
        max_jobs = 4
        ret = luigi.build([RegionFeaturesWorkflow(input_path=self.input_path,
                                                  input_key=self.input_key,
                                                  labels_path=self.input_path,
                                                  labels_key=self.seg_key,
                                                  output_path=self.output_path,
                                                  output_key=self.output_key,
                                                  config_dir=self.config_folder,
                                                  tmp_folder=self.tmp_folder,
                                                  target=self.target,
                                                  max_jobs=max_jobs)],
                          local_scheduler=True)
        self.assertTrue(ret)
        self._check_subresults()
        self._check_result()


if __name__ == '__main__':
    unittest.main()
