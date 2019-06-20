#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python

import os
import json
import luigi

from .. import MulticutSegmentationWorkflow


def run_mc(sample, tmp_folder, max_jobs,
           n_scales=1, have_watershed=True, target='local',
           from_affinities=False, invert_inputs=False):
    """ Run multicut on cremi sample or similar data.

    You can obtain the data used for this examle from
    https://drive.google.com/open?id=1E6j77gV0iwquSxd7KmmuXghgFcyuP7WW

    Args:
        sample: which cremi sample to use (more general, what's our input data)
        tmp_folder: temporary folder to store job files
        max_jobs: maximal number of jobs
        n_scales: number of scales for hierarchical solver (0 will perform vanilla multicut)
        have_watershed: flag to indicate if the watershed is computed already
        target: target platform, either 'local' (computation on local host),
                                        'slurm' (cluster running slurm)
                                     or 'lsf' (cluster running lsf)
        from_affinities: whether to use affinity maps or boundary maps
        invert_inputs: whether to invert the inputs; this needs to be set to true
            if HIGH boundary evidence correponds to LOWER values in boundary /
            affinity maps
    """

    # input path: n5 or hdf5 container which holds the input data
    # (= boundary maps or affinity maps)
    input_path = '/g/kreshuk/data/cremi/example/sample%s.n5' % sample
    # path with the watershed data, can be the same as input_path
    ws_path = input_path

    # key for input, and watershed
    input_key = 'predictions/full_affs'
    ws_key = 'segmentation/watershed'

    # path to n5 or hdf5 container to which the output segmentation should be written
    # can be the same as input_path
    out_path = input_path
    out_key = 'segmentation/multicut'

    # path and key for mask
    # mask can be used to exclude parts of the volume from segmentation
    # leave blank if you don't have a mask
    mask_path = ''
    mask_key = ''

    # n5 container for intermediate results like graph-structure or features
    exp_path = './sample%s_exp.n5' % sample

    # config folder holds configurations for workflow steps stored as json
    configs = MulticutSegmentationWorkflow.get_config()
    config_folder = 'config_mc'
    os.makedirs(config_folder, exist_ok=True)

    # global workflow config
    # python interpreter of conda environment with dependencies, see
    # https://github.com/constantinpape/cluster_tools/blob/master/environment.yml
    shebang = "#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python"
    # block shape used for parallelization
    block_shape = [30, 256, 256]
    global_config = configs['global']
    global_config.update({'shebang': shebang, 'block_shape': block_shape})
    with open('./config_mc/global.config', 'w') as f:
        json.dump(global_config, f)

    # config for edge feature calculation
    feat_config = configs['block_edge_features']
    # specify offsets if you have affinity features.
    if from_affinities:
        feat_config.update({'offsets': [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]})
    with open('./config_mc/block_edge_features.config', 'w') as f:
        json.dump(feat_config, f)

    # config for converting edge probabilities to edge costs
    costs_config = configs['probs_to_costs']
    costs_config.update({'threads_per_job': max_jobs, 'weight_edges': True, 'invert_inputs': invert_inputs})
    with open('./config_mc/probs_to_costs.config', 'w') as f:
        json.dump(costs_config, f)

    # set number of threads for sum jobs
    tasks = ['merge_sub_graphs', 'merge_edge_features', 'probs_to_costs',
             'solve_subproblems', 'reduce_problem', 'solve_global']
    for tt in tasks:
        config = configs[tt]
        config.update({'threads_per_job': max_jobs})
        with open('./config_mc/%s.config' % tt, 'w') as f:
            json.dump(config, f)

    luigi.build([MulticutSegmentationWorkflow(input_path=input_path, input_key=input_key,
                                              ws_path=ws_path, ws_key=ws_key,
                                              mask_path=mask_path, mask_key=mask_key,
                                              problem_path=exp_path,
                                              node_labels_key='node_labels',
                                              output_path=out_path, output_key=out_key,
                                              n_scales=n_scales,
                                              config_dir='./config_mc',
                                              tmp_folder=tmp_folder,
                                              target=target,
                                              skip_ws=have_watershed,
                                              max_jobs=max_jobs)], local_scheduler=True)


if __name__ == '__main__':
    sample = 'A'
    tmp_folder = './tmp_mc_%s' % sample

    target = 'slurm'
    max_jobs = 16

    run_mc(sample, tmp_folder, max_jobs, target=target, from_affinities=True)
