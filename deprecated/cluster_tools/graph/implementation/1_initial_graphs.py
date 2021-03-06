#! /usr/bin/python

import os
import time
import argparse
import numpy as np

import z5py
import nifty
import nifty.distributed as ndist


def extract_subgraph_from_roi(block_id, blocking, labels_path, labels_key, graph_path):
    halo = [1, 1, 1]
    block = blocking.getBlockWithHalo(block_id, halo)
    outer_block, inner_block = block.outerBlock, block.innerBlock
    # we only need the halo into one direction,
    # hence we use the outer-block only for the end coordinate
    begin = inner_block.begin
    end = outer_block.end

    block_key = 'sub_graphs/s0/block_%i' % block_id
    ndist.computeMergeableRegionGraph(labels_path, labels_key,
                                      begin, end,
                                      graph_path, block_key)


def graph_step1(labels_path, labels_key, graph_path, block_file, block_shape):
    t0 = time.time()
    labels = z5py.File(labels_path)[labels_key]
    shape = labels.shape
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(block_shape))
    block_list = np.load(block_file)

    for block_id in block_list:
        extract_subgraph_from_roi(block_id, blocking, labels_path, labels_key, graph_path)

    job_id = int(os.path.split(block_file)[1].split('_')[2][:-4])
    print("Success job %i" % job_id)
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_path", type=str)
    parser.add_argument("labels_key", type=str)
    parser.add_argument("graph_path", type=str)
    parser.add_argument("--block_file", type=str)
    parser.add_argument("--block_shape", nargs=3, type=int)
    args = parser.parse_args()

    graph_step1(args.labels_path, args.labels_key,
                args.graph_path, args.block_file,
                list(args.block_shape))
