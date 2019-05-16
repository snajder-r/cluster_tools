#! /bin/python

import os
import sys
import json
import pickle

import numpy as np
import luigi
from sklearn.ensemble import RandomForestClassifier
import itertools
import sklearn.metrics

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Learning Tasks
#

# TODO implement graph extraction with ignore label 0
class LearnRFBase(luigi.Task):
    """ LearnRF base class
    """

    task_name = 'learn_rf'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input volumes and graph
    features_dict = luigi.DictParameter()
    labels_dict = luigi.DictParameter()
    output_path = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'n_trees': 100})
        return config

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()
        assert self.features_dict.keys() == self.labels_dict.keys()

        # NOTE we have to turn the luigi dict parameters into normal python dicts
        # in order to json serialize them
        config.update({'features_dict': {key: val for key, val in self.features_dict.items()},
                       'labels_dict': {key: val for key, val in self.labels_dict.items()},
                       'output_path': self.output_path})

        # prime and run the jobs
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class LearnRFLocal(LearnRFBase, LocalTask):
    """ LearnRF on local machine
    """
    pass


class LearnRFSlurm(LearnRFBase, SlurmTask):
    """ LearnRF on slurm cluster
    """
    pass


class LearnRFLSF(LearnRFBase, LSFTask):
    """ LearnRF on lsf cluster
    """
    pass


#
# Implementation
#


def learn_rf(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    features_dict = config['features_dict']
    labels_dict = config['labels_dict']
    output_path = config['output_path']
    n_threads = config['threads_per_job']
    n_trees = config.get('n_trees', 200)

    features = []
    val_features = []
    labels = []
    val_labels = []

    feature_colnames = []

    # TODO enable multiple feature paths
    # NOTE we assert that keys of boyh dicts are identical in the main class
    for key, feat_path in features_dict.items():
        label_path = labels_dict[key]
        fu.log("reading featurs from %s:%s, labels from %s:%s" % tuple(feat_path + label_path))

        with vu.file_reader(feat_path[0]) as f:
            ds = f[feat_path[1]]
            ds.n_threads = n_threads
            feats = ds[:]
            feature_colnames = ds.attrs['feature_colnames']

            assert len(feature_colnames) == feats.shape[1]

        with vu.file_reader(label_path[0]) as f:
            ds = f[label_path[1]]
            ds.n_threads = n_threads
            label = ds[:]
        assert len(label) == len(feats)

        # check if we have an ignore label
        ignore_mask = label != -1
        n_ignore = np.sum(ignore_mask)
        if n_ignore < ignore_mask.size:
            fu.log("removing %i examples due to ignore mask" % n_ignore)
            feats = feats[ignore_mask]
            label = label[ignore_mask]

        if key.startswith('validation_'):
            val_features.append(feats)
            val_labels.append(label)
        else:
            features.append(feats)
            labels.append(label)

    
    features = np.concatenate(features, axis=0)
    val_features = np.concatenate(val_features, axis=0)
    labels = np.concatenate(labels, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    val_num_de = (val_labels).sum()
    val_weight_de = len(val_labels) / val_num_de
    val_weight_other = len(val_labels)/(len(val_labels)-val_num_de)
    val_sample_weights = np.ones(val_labels.shape, dtype='float32') * (val_labels==1) * val_weight_de
    val_sample_weights[val_labels==0] = val_weight_other

    train_num_de = (labels).sum()
    train_weight_de = len(labels) / train_num_de
    train_weight_other = len(labels)/(len(labels)-train_num_de)
    train_sample_weights = np.ones(labels.shape, dtype='float32') * (labels==1) * train_weight_de
    train_sample_weights[labels==0] = train_weight_other
    
    fu.log("start learning random forest with %i examples and %i features" % features.shape)
    fu.log("validation set has %d double edges" % val_num_de)
    fu.log("Training set has %d double edges" % train_num_de)

    unique_features = set(feature_colnames)
    fu.log("Unique features: %s" % unique_features)    
    best_error = 10000
    best_features = None
    best_prediction = None

    for n_trees in [75,100]:
        for num_features in range(3,len(unique_features)-1):
            for feature_selection in itertools.combinations(unique_features,num_features):
                for attempt in range(2):
                    selection_indices, = np.where([fv in feature_selection for fv in feature_colnames])
                    print("Feature names: ", feature_colnames)
                    print("Subfeatures: ", selection_indices)
                    subfeatures = features[:,selection_indices]
                    val_subfeatures = val_features[:,selection_indices]

                    rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=n_threads, class_weight="balanced")
                    rf.fit(subfeatures, labels)
 
                    val_prediction = rf.predict_proba(val_subfeatures)
                    train_prediction = rf.predict_proba(subfeatures)
     
                    val_error = sklearn.metrics.log_loss(val_labels, val_prediction, sample_weight = val_sample_weights)
                    train_error = sklearn.metrics.log_loss(labels, train_prediction, sample_weight = train_sample_weights)

                    fu.log('Train error: %f   Val error: %f' %(train_error, val_error))
                    if val_error < best_error:
                        fu.log("New best Features: %s" % str(feature_selection))
                        fu.log("Predicts %d double edges in validation set " % (val_prediction[:,1]>0.5).sum())
                        fu.log("Predicts %d double edges in training set " % (train_prediction[:,1]>0.5).sum())
                        fu.log("Ntrees: %d" % n_trees)
                        fu.log("saving random forest to %s" % output_path)
                        with open(output_path, 'wb') as f:
                            pickle.dump(rf, f)

                        best_features = feature_selection
                        best_prediction = train_prediction.copy()
                        best_error = val_error
                    else: 
                        fu.log("not as good %f %s" % (val_error, feature_selection))


    print("Errors on training set: ", (len(labels) - np.sum((best_prediction[:,1]>0.5).astype('uint8') == labels)))
    

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    learn_rf(job_id, path)
