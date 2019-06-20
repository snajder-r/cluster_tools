#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import sys
import json

import luigi
import h5py
import z5py

from ...graph import GraphWorkflow
from ...watershed.watershed import WatershedLocal


def graph_example(shebang):
    input_path = '/home/cpape/Work/data/isbi2012/cluster_example/isbi_train.n5'
    input_key = 'volumes/watersheds'
    output_path = '/home/cpape/Work/data/isbi2012/cluster_example/graph.n5'

    tmp_folder = './tmp'
    config_folder = './configs'

    max_jobs = 8
    global_conf = WatershedLocal.default_global_config()
    global_conf.update({'shebang': shebang, 'block_shape': [10, 256, 256]})
    with open('./configs/global.config', 'w') as f:
        json.dump(global_conf, f)

    ret = luigi.build([GraphWorkflow(input_path=input_path,
                                     input_key=input_key,
                                     graph_path=output_path,
                                     n_scales=1,
                                     config_dir=config_folder,
                                     tmp_folder=tmp_folder,
                                     target='local',
                                     max_jobs=max_jobs)], local_scheduler=True)

if __name__ == '__main__':
    shebang = '#! /home/cpape/Work/software/conda/miniconda3/envs/affogato/bin/python'
    graph_example(shebang)
