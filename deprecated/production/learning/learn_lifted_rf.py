import os
import pickle
import numpy as np

import nifty
import nifty.graph.rag as nrag
# import nifty.graph.opt.lifted_multicut as nlmc

import z5py
from sklearn.ensemble import RandomForestClassifier

from .. import features as feat


def extract_feats_and_labels(path, aff_key, ws_key, gt_key, mask_key, lifted_nh,
                             offsets=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
                             n_threads=40):
    f = z5py.File(path)

    # load the watershed segmentation and compute rag
    ds_seg = f[ws_key]
    ds_seg.n_threads = n_threads
    seg = ds_seg[:]
    print(seg.shape)
    n_labels = int(seg.max()) + 1
    rag = nrag.gridRag(seg, numberOfLabels=n_labels,
                       numberOfThreads=n_threads)

    # load affinities and glia channel
    ds_affs = f[aff_key]
    ds_affs.n_threads = n_threads
    aff_slice = slice(0, len(offsets))
    affs = ds_affs[aff_slice]
    if affs.dtype == np.dtype('uint8'):
        affs = affs.astype('float32') / 255.
    affs = 1. - affs

    n_chans = ds_affs.shape[0]
    glia_slice = slice(n_chans - 1, n_chans)
    glia = ds_affs[glia_slice]
    if glia.dtype == np.dtype('uint8'):
        glia = glia.astype('float32') / 255.

    # compute local probs from affinities
    print("Computing local probabilities")
    probs = nrag.accumulateAffinityStandartFeatures(rag, affs, offsets,
                                                    numberOfThreads=n_threads)[:, 0]
    probs = np.nan_to_num(probs)

    # remove zero-label (== ignore label) from the graph, because it short-circuits
    # lifted edges
    uv_ids = rag.uvIds()
    valid_edges = (uv_ids != 0).all(axis=1)
    uv_ids = uv_ids[valid_edges]
    probs = probs[valid_edges]

    # compute the lifted graph and lifted features
    print("Computing lifted objective")
    lifted_uv_ids = feat.make_filtered_lifted_nh(rag, n_labels, uv_ids, lifted_nh)
    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uv_ids)

    # TODO parallelize some of these
    print("Computing lifted features")
    features = np.concatenate([  # feat.ucm_features(n_labels, lifted_objective, probs),
                               feat.clustering_features(graph, probs, lifted_uv_ids),
                               feat.ucm_features(n_labels, uv_ids, lifted_uv_ids, probs),
                               feat.region_features(seg, lifted_uv_ids, glia)], axis=1)

    # load mask and groundtruth
    ds_mask = f[mask_key]
    ds_mask.n_threads = n_threads
    mask = ds_mask[:]

    ds_gt = f[gt_key]
    ds_gt.n_threads = n_threads
    gt = ds_gt[:]
    gt[np.logical_not(mask)] = 0

    # compute the edge labels and valid edges
    node_labels = nrag.gridRagAccumulateLabels(rag, gt)
    labels = (node_labels[lifted_uv_ids[:, 0]] != node_labels[lifted_uv_ids[:, 1]]).astype('uint8')
    valid_edges = (node_labels[lifted_uv_ids] != 0).all(axis=1)
    print(np.sum(valid_edges), "edges of", len(lifted_uv_ids), "are valid")
    assert features.shape[0] == labels.shape[0]

    # just for temporary inspection, deactivate !
    import vigra
    vigra.writeHDF5(features, './feats_tmp.h5', 'data', chunks=True)
    vigra.writeHDF5(labels, './labs_tmp.h5', 'data', chunks=True)

    return features[valid_edges], labels[valid_edges]


def learn_lifted_rf(paths, save_path, lifted_nh,
                    aff_key='volumes/predictions/affinities',
                    ws_key='volumes/labels/watershed',
                    gt_key='volumes/labels/neuron_ids',
                    mask_key='volumes/labels/mask',
                    offsets=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
                    n_threads=40, n_trees=200,
                    max_depth=None):
    assert all(os.path.exists(path) for path in paths)
    features, labels = [], []
    for path in paths:
        print("Computing lifted features and labels from", path)
        feats, labs = extract_feats_and_labels(path, aff_key, ws_key,
                                               gt_key, mask_key, lifted_nh,
                                               offsets=offsets,
                                               n_threads=n_threads)
        features.append(feats)
        labels.append(labs)
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    assert len(features) == len(labels)

    print("Start fitting rf ...")
    rf = RandomForestClassifier(n_jobs=n_threads, n_estimators=n_trees,
                                class_weight='balanced', max_depth=max_depth)
    rf.fit(features, labels)
    print("... done")
    rf.n_jobs = 1
    with open(save_path, 'wb') as f:
        pickle.dump(rf, f)
