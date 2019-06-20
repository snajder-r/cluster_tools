import os
import sys

EXECUTABLE = '/groups/saalfeld/home/papec/Work/software/conda/miniconda3/envs/production/bin/python'


def make_ws_scripts(path, n_jobs, block_shape, tmp_dir):
    sys.path.append('../../..')
    from ...masked_watershed import make_batch_jobs
    chunks = [bs // 2 for bs in block_shape]
    # chunks = block_shape
    make_batch_jobs(path, 'predictions/affs_glia',
                    path, 'masks/minfilter_mask',
                    path, 'segmentations/watershed2',
                    os.path.join(tmp_dir, 'tmp_files', 'tmp_ws'),
                    block_shape, chunks, n_jobs, EXECUTABLE,
                    use_bsub=True,
                    n_threads_ufd=4,
                    eta=[20, 5, 5, 5])


def make_relabel_scripts(path, n_jobs, block_shape, tmp_dir):
    sys.path.append('../../..')
    from ...relabel import make_batch_jobs
    make_batch_jobs(path, 'segmentations/dtws',
                    os.path.join(tmp_dir, 'tmp_files', 'tmp_relabel'),
                    block_shape, n_jobs,
                    EXECUTABLE,
                    use_bsub=True,
                    eta=[5, 5, 5])


def make_graph_scripts(path, n_scales, n_jobs, n_threads, block_shape, tmp_dir):
    sys.path.append('../../..')
    from ...graph import make_batch_jobs
    make_batch_jobs(path, 'segmentations/dtws',
                    os.path.join(tmp_dir, 'tmp_files', 'graph.n5'),
                    os.path.join(tmp_dir, 'tmp_files', 'tmp_graph'),
                    block_shape,
                    n_scales, n_jobs,
                    EXECUTABLE,
                    use_bsub=True,
                    n_threads_merge=n_threads,
                    eta=[10, 10, 10, 10])


def make_feature_scripts(path, n_jobs1, n_jobs2, n_threads, block_shape, tmp_dir):
    sys.path.append('../../..')
    from ...features import make_batch_jobs
    make_batch_jobs(os.path.join(tmp_dir, 'tmp_files', 'graph.n5'), 'graph',
                    os.path.join(tmp_dir, 'tmp_files', 'features.n5'), 'features',
                    path, 'predictions/affs_glia',
                    path, 'segmentations/dtws',
                    os.path.join(tmp_dir, 'tmp_files', 'tmp_features'),
                    block_shape,
                    n_jobs1, n_jobs2,
                    # n_threads2=n_threads,
                    n_threads2=8,
                    executable=EXECUTABLE,
                    use_bsub=True,
                    eta=[20, 5])


def make_cost_scripts(path, n_jobs, n_threads, tmp_dir, rf_path):
    sys.path.append('../../..')
    from ...costs import make_batch_jobs
    make_batch_jobs(os.path.join(tmp_dir, 'tmp_files', 'features.n5'), 'features',
                    os.path.join(tmp_dir, 'tmp_files', 'graph.n5'), 'graph',
                    rf_path,
                    os.path.join(tmp_dir, 'tmp_files', 'costs.n5'), 'costs',
                    n_jobs,
                    os.path.join(tmp_dir, 'tmp_files', 'tmp_costs'),
                    n_threads,
                    executable=EXECUTABLE,
                    use_bsub=True,
                    eta=5)


def make_multicut_scripts(path, n_scales, n_threads, block_shape, tmp_dir, res_key):
    sys.path.append('../../..')
    n_jobs = 12
    from ...multicut import make_batch_jobs
    make_batch_jobs(os.path.join(tmp_dir, 'tmp_files', 'graph.n5'), 'graph',
                    os.path.join(tmp_dir, 'tmp_files', 'costs.n5'), 'costs',
                    path, 'node_labelings/%s' % res_key,
                    block_shape, n_scales,
                    os.path.join(tmp_dir, 'tmp_files', 'tmp_mc'),
                    n_jobs,
                    n_threads=n_threads,
                    executable=EXECUTABLE,
                    use_bsub=True,
                    eta=[5, 5, 15])


def make_projection_scripts(path, n_jobs, block_shape, tmp_dir, res_key):
    sys.path.append('../../..')
    from ...label_projection import make_batch_jobs
    # chunks = [bs // 2 for bs in block_shape]
    chunks = block_shape
    make_batch_jobs(path, 'segmentations/dtws',
                    path, 'segmentations/%s' % res_key,
                    path, 'node_labelings/%s' % res_key,
                    os.path.join(tmp_dir, 'tmp_files', 'tmp_projection'),
                    block_shape, chunks, n_jobs,
                    executable=EXECUTABLE,
                    use_bsub=True,
                    eta=5)


def make_scripts(sample,
                 n_scales,
                 n_jobs_max,
                 n_threads_max,
                 block_shape,
                 tmp_dir,
                 rf_path=''):

    path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample%s.n5' % sample

    if rf_path == '':
        res_key = 'mc_new_dt'
    else:
        res_key = 'mc_glia_rf_affs_global'

    # make folders
    if not os.path.exists(os.path.join(path, 'segmentations')):
        os.mkdir(os.path.join(path, 'segmentations'))
    if not os.path.exists(os.path.join(path, 'node_labelings')):
        os.mkdir(os.path.join(path, 'node_labelings'))
    if not os.path.exists(os.path.join(tmp_dir, 'tmp_files')):
        os.mkdir(os.path.join(tmp_dir, 'tmp_files'))

    # make the ws scripts
    if not os.path.exists('./1_watershed'):
        os.mkdir('./1_watershed')
    os.chdir('./1_watershed')
    make_ws_scripts(path, n_jobs_max, block_shape, tmp_dir)
    os.chdir('..')

    # make the relabeling scripts
    if not os.path.exists('./2_relabel'):
        os.mkdir('./2_relabel')
    os.chdir('./2_relabel')
    make_relabel_scripts(path, n_jobs_max, block_shape, tmp_dir)
    os.chdir('..')

    # make the graph scripts
    if not os.path.exists('./3_graph'):
        os.mkdir('./3_graph')
    os.chdir('./3_graph')
    make_graph_scripts(path, n_scales, n_jobs_max, n_threads_max, block_shape, tmp_dir)
    os.chdir('..')

    # make the feature scripts
    if not os.path.exists('./4_features'):
        os.mkdir('./4_features')
    os.chdir('./4_features')
    make_feature_scripts(path, n_jobs_max, 1, n_threads_max, block_shape, tmp_dir)
    os.chdir('..')

    # make the costs scripts
    if not os.path.exists('./4a_costs'):
        os.mkdir('./4a_costs')
    os.chdir('./4a_costs')
    make_cost_scripts(path, 4, 12, tmp_dir, rf_path)
    os.chdir('..')

    # make the multicut scripts
    if not os.path.exists('./5_multicut'):
        os.mkdir('./5_multicut')
    os.chdir('./5_multicut')
    # 100  jobs is way too much
    make_multicut_scripts(path, n_scales, n_threads_max, block_shape, tmp_dir, res_key)
    os.chdir('..')

    # make the projection scripts
    if not os.path.exists('./6_label_projection'):
        os.mkdir('./6_label_projection')
    os.chdir('./6_label_projection')
    make_projection_scripts(path, n_jobs_max, block_shape, tmp_dir, res_key)
    os.chdir('..')


if __name__ == '__main__':
    sample = 'A+'
    tmp_dir = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cache/cremi_%s_dt' % sample
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    n_jobs = 128
    n_scales = 1
    n_threads = 12
    block_shape = (50, 512, 512)

    rf_path = ''
    # rf_path = '/nrs/saalfeld/papec/cremi2.0/training_data/V1_20180419/rf_default_affs.pkl'

    make_scripts(sample, n_scales, n_jobs, n_threads, block_shape, tmp_dir, rf_path)
