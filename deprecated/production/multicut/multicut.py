#! /usr/bin/python

import os
import time
import argparse
import json
import pickle
import subprocess

import numpy as np
import nifty
import nifty.graph.rag as nrag
import z5py
import luigi
from production import multicut
from production import features as feat


# TODO computation with rois
class MulticutTask(luigi.Task):
    """
    Run all thresholding tasks
    """

    # path to the n5 file and keys
    path = luigi.Parameter()
    aff_key = luigi.Parameter()
    ws_key = luigi.Parameter()
    out_key = luigi.Parameter()
    # dummy parameter to be consistent with other segmentation tasks
    max_jobs = luigi.IntParameter()
    # path to the configuration
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        return self.dependency

    def _collect_outputs(self):
        res_path = os.path.join(self.tmp_folder, 'multicut_time.json')
        try:
            assert os.path.exists(res_path)
            with open(res_path) as f:
                t = json.load(f)['t']
            os.remove(res_path)
        except Exception:
            return None
        return t

    def run(self):
        from .. import util

        # copy the script to the temp folder and replace the shebang
        script_path = os.path.join(self.tmp_folder, 'multicut.py')
        file_dir = os.path.dirname(os.path.abspath(__file__))
        util.copy_and_replace(os.path.join(file_dir, 'multicut.py'), script_path)

        with open(self.config_path) as f:
            config = json.load(f)
            if 'roi' in config:
                have_roi = True

        assert have_roi

        config_path = os.path.join(self.tmp_folder, 'multicut_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        command = '%s %s %s %s %s %s %s' % (script_path, self.path, self.aff_key, self.ws_key,
                                            self.out_key, self.tmp_folder, config_path)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_multicut')
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_multicut.err')
        bsub_command = 'bsub -J multicut -We %i -o %s -e %s \'%s\'' % (self.time_estimate,
                                                                       log_file, err_file, command)
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)

        # wait till all jobs are finished
        if not self.run_local:
            util.wait_for_jobs('papec')
        t = self._collect_outputs()
        success = t is not None

        out_path = self.output().path
        success = os.path.exists(out_path)

        if not success:
            raise RuntimeError("MulticutTask failed")

    def output(self):
        return luigi.LocalTarget(os.path.join(self.path, self.out_key))


def _mc_impl(ws, affs, offsets,
             weight_mulitcut_edges=False,
             weighting_exponent=1):

    ws = ws.astype('uint32')
    n_labels = int(ws.max()) + 1
    rag = nrag.gridRag(ws,
                       numberOfLabels=n_labels,
                       numberOfThreads=8)
    uv_ids = rag.uvIds()

    # compute the features and get edge probabilities (from mean affinities)
    # and edge sizes
    features = nrag.accumulateAffinityStandartFeatures(rag, affs, offsets,
                                                       numberOfThreads=8)
    probs = features[:, 0]
    sizes = features[:, -1]

    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uv_ids)
    # compute multicut edge results
    node_labels, _ = multicut.run_mc(graph, probs, uv_ids,
                                     with_ignore_edges=True,
                                     edge_sizes=sizes if weight_mulitcut_edges else None,
                                     weighting_exponent=weighting_exponent)
    return node_labels


def _mc_learned_impl(ws, affs, rfs,
                     weight_mulitcut_edges=False,
                     weighting_exponent=1):

    assert len(rfs) == 2
    rf_xy, rf_z = rfs

    ws = ws.astype('uint32')
    n_labels = int(ws.max()) + 1
    rag = nrag.gridRag(ws,
                       numberOfLabels=n_labels,
                       numberOfThreads=8)
    uv_ids = rag.uvIds()

    # TODO add glia features ?
    features, sizes, z_edges = feat.edge_features(rag, ws, n_labels, uv_ids, affs[:3],
                                                  n_threads=8)

    probs = np.zeros(len(features))
    xy_edges = np.logical_not(z_edges)
    if np.sum(xy_edges) > 0:
        probs[xy_edges] = rf_xy.predict_proba(features[xy_edges])[:, 1]
    if np.sum(z_edges) > 0:
        probs[z_edges] = rf_z.predict_proba(features[z_edges])[:, 1]

    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uv_ids)
    # compute multicut edge results
    node_labels, _ = multicut.run_mc(graph, probs, uv_ids,
                                     with_ignore_edges=True,
                                     edge_sizes=sizes if weight_mulitcut_edges else None,
                                     weighting_exponent=weighting_exponent)
    return node_labels


def single_multicut(path, aff_key, ws_key, out_key,
                    tmp_folder, config_path):

    from production.util import normalize_and_save_assignments
    t0 = time.time()
    # load the blocks to be processed and the configuration from the input config file
    with open(config_path) as f:
        config = json.load(f)
    offsets = config['affinity_offsets']
    weight_mulitcut_edges = config['weight_multicut_edges']
    weighting_exponent = config.get('weighting_exponent', 1.)
    roi = config['roi']
    rf_path = config.get('rf_path', None)
    n_threads = config['n_threads']

    # TODO support lmc
    # use_lifted = config.get('use_lifted', False)
    # lifted_nh = config.get('lifted_nh', None)

    # open all n5 datasets
    ds_ws = z5py.File(path)[ws_key]
    # hack for the threads ...
    ds_ws.n_threads = n_threads
    ds_affs = z5py.File(path)[aff_key]
    ds_affs.n_threads = n_threads

    # get the bounding box and load affinities and
    # watershed
    bb = tuple(slice(beg, end) for beg, end in zip(roi[0], roi[1]))
    ws = ds_ws[bb]

    # load the affinities
    n_channels = len(offsets)
    bb_affs = (slice(0, n_channels),) + bb
    affs = ds_affs[bb_affs]
    # convert affinities to float and invert them
    # to get boundary probabilities
    if affs.dtype == np.dtype('uint8'):
        affs = affs.astype('float32') / 255.
    affs = 1. - affs

    if rf_path is not None:
        assert len(rf_path) == 2
        rf = []
        with open(rf_path[0], 'rb') as f:
            rf.append(pickle.load(f))
        with open(rf_path[1], 'rb') as f:
            rf.append(pickle.load(f))
    else:
        rf = None

    # compute the multicut result
    if rf is None:
        node_labels = _mc_impl(ws, affs, offsets,
                               weight_mulitcut_edges=weight_mulitcut_edges,
                               weighting_exponent=weighting_exponent)
    else:
        node_labels = _mc_learned_impl(ws, affs, rf,
                                       weight_mulitcut_edges=False,
                                       weighting_exponent=1)

    normalize_and_save_assignments(path, out_key, node_labels, n_threads)

    out_path = os.path.join(tmp_folder, 'multicut_time.json')
    with open(out_path, 'w') as f:
        json.dump({'t': time.time() - t0}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('aff_key', type=str)
    parser.add_argument('ws_key', type=str)
    parser.add_argument('out_key', type=str)
    parser.add_argument('tmp_folder', type=str)
    parser.add_argument('config_path', type=str)

    args = parser.parse_args()
    single_multicut(args.path,
                    args.aff_key,
                    args.ws_key,
                    args.out_key,
                    args.tmp_folder,
                    args.config_path)
