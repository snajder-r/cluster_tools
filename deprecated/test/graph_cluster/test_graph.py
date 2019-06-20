import sys
sys.path.append('../..')
from ...graph import make_batch_jobs

LABELS_PATH = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cluster_test_data/testdata1.n5'

LABELS_KEY = 'segmentations/watershed'

GRAPH_PATH = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cluster_test_data/graph.n5'

TMP_FOLDER = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cluster_test_data/tmp_files_graph'
BLOCK_SHAPE = (25, 256, 256)
EXECUTABLE = '/groups/saalfeld/home/papec/Work/software/conda/miniconda3/envs/production/bin/python'


def jobs_for_cluster_test(n_jobs, n_scales):
    make_batch_jobs(LABELS_PATH, LABELS_KEY,
                    GRAPH_PATH, TMP_FOLDER,
                    BLOCK_SHAPE,
                    n_scales, n_jobs,
                    EXECUTABLE,
                    use_bsub=True,
                    n_threads_merge=4,
                    eta=[5, 5, 5, 5])


if __name__ == '__main__':
    n_jobs = 32
    n_scales = 1
    jobs_for_cluster_test(n_jobs, n_scales)
