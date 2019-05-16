import os
import json
import luigi

import cluster_tools.utils.volume_utils as vu
from ..cluster_tasks import WorkflowBase
# TODO Region features
from ..features import EdgeFeaturesWorkflow, EdgeFeaturesWorkflowWithRegionFeatures, RegionFeaturesWorkflow
from ..graph import GraphWorkflow
from ..node_labels import NodeLabelWorkflow
from . import edge_labels_mc as label_tasks_mc
from . import node_labels_de as label_tasks_de
from . import conseq_labels as conseq_labels
from . import learn_rf as learn_tasks
from . import merge_features as merge_tasks


def read_number_of_labels(path):
    try:
        with vu.file_reader(path[0], 'r') as f:
            n_labels = f[path[1]].attrs['maxId'] + 1
        return int(n_labels)
    except:
        print("Can't read maxId in %s/%s" % (path[0],path[1]))
        raise



class LearningWorkflowPreparationTaskMC(WorkflowBase):
    input_dict = luigi.DictParameter()
    probs_dict = luigi.DictParameter()
    labels_dict = luigi.DictParameter()
    groundtruth_dict = luigi.DictParameter()
    region_features = luigi.ListParameter()
    dependency = luigi.TaskParameter()

    
    def _check_input(self):
        assert self.input_dict.keys() == self.labels_dict.keys()
        assert self.input_dict.keys() == self.probs_dict.keys()
        assert self.input_dict.keys() == self.groundtruth_dict.keys()

    def requires(self):
        self._check_input()
        n_scales = 1

        try:
            os.mkdir(self.tmp_folder)
        except OSError:
            pass

        prev_dep = self.dependency
        task_list = list()
        for key, input_path in self.input_dict.items():
            probs_path = self.probs_dict[key]
            labels_path = self.labels_dict[key]
            gt_path = self.groundtruth_dict[key]
            number_of_labels = read_number_of_labels(labels_path)

            # we need different tmp folders for each input dataset
            tmp_folder = os.path.join(self.tmp_folder, key)

            graph_out = os.path.join(tmp_folder, 'graph_ws.n5')
            graph_key = 'graph'
            graph_task = GraphWorkflow(tmp_folder=tmp_folder,
                                       max_jobs=self.max_jobs,
                                       config_dir=self.config_dir,
                                       target=self.target,
                                       dependency=prev_dep,
                                       input_path=labels_path[0],
                                       input_key=labels_path[1],
                                       graph_path=graph_out,
                                       output_key=graph_key,
                                       n_scales=n_scales)

            features_out = os.path.join(tmp_folder, 'features.n5')
            
            feat_task = EdgeFeaturesWorkflowWithRegionFeatures(tmp_folder=tmp_folder,
                                             max_jobs=self.max_jobs,
                                             config_dir=self.config_dir,
                                             dependency=graph_task,
                                             target=self.target,
                                             input_paths=input_path[0],
                                             input_keys=input_path[1],
                                             probs_path=probs_path[0],
                                             probs_key=probs_path[1],
                                             labels_path=labels_path[0],
                                             labels_key=labels_path[1],
                                             graph_path=graph_out,
                                             graph_key=graph_key,
                                             output_path=features_out,
                                             output_key='features',
                                             number_of_labels=-1,
                                             region_features=self.region_features,
                                             use_edge_features=True)
            
            node_labels_out = os.path.join(tmp_folder, 'gt_node_labels.n5')
            node_labels_task = NodeLabelWorkflow(tmp_folder=tmp_folder,
                                            max_jobs=self.max_jobs,
                                            config_dir=self.config_dir,
                                            target=self.target,
                                            dependency=feat_task,
                                            ws_path=labels_path[0],
                                            ws_key=labels_path[1],
                                            input_path=gt_path[0],
                                            input_key=gt_path[1],
                                            output_path=node_labels_out,
                                            output_key='node_labels')

            edge_labels_out = os.path.join(tmp_folder, 'edge_labels.n5')
            lt = getattr(label_tasks_mc,
                         self._get_task_name('EdgeLabels'))
            label_task = lt(tmp_folder=tmp_folder,
                            max_jobs=self.max_jobs,
                            config_dir=self.config_dir,
                            dependency=node_labels_task,
                            graph_path=graph_out,
                            graph_key='graph',
                            node_labels_path=node_labels_out,
                            node_labels_key='node_labels',
                            ws_path=labels_path[0],
                            ws_key=labels_path[1],
                            output_path=edge_labels_out,
                            output_key='edge_labels')

            yield label_task
     


class LearningWorkflowPreparationTaskDE(WorkflowBase):
    input_dict = luigi.DictParameter()
    probs_dict = luigi.DictParameter()
    groundtruth_dict = luigi.DictParameter()
    groundtruth_sem_dict = luigi.DictParameter()
    region_features = luigi.ListParameter()
    dependency = luigi.TaskParameter()

    
    def _check_input(self):
        assert self.input_dict.keys() == self.probs_dict.keys()
        assert self.input_dict.keys() == self.groundtruth_dict.keys()
        assert self.input_dict.keys() == self.groundtruth_sem_dict.keys()

    def requires(self):
        self._check_input()
        n_scales = 1

        try:
            os.mkdir(self.tmp_folder)
        except OSError:
            pass

        prev_dep = self.dependency
        task_list = list()
        for key, input_path in self.input_dict.items():
            probs_path = self.probs_dict[key]
            gt_path_ori = self.groundtruth_dict[key]
            gt_sem_path = self.groundtruth_sem_dict[key]
            # we need different tmp folders for each input dataset
            tmp_folder = os.path.join(self.tmp_folder, key)
            gt_path = (os.path.join(tmp_folder, 'gt_conseq.n5'), gt_path_ori[1])

            number_of_labels = read_number_of_labels(gt_path_ori)

            ct = getattr(conseq_labels, self._get_task_name('ConsequtiveLabels'))
            conseq_task = ct(input_path=gt_path_ori[0], input_key=gt_path_ori[1], 
                output_path=gt_path[0], output_key=gt_path[1],
                dependency=prev_dep,tmp_folder=tmp_folder,
                max_jobs=self.max_jobs, config_dir=self.config_dir)

            graph_out = os.path.join(tmp_folder, 'graph_gt.n5')
            graph_key = 'graph'
            graph_task = GraphWorkflow(tmp_folder=tmp_folder,
                                       max_jobs=self.max_jobs,
                                       config_dir=self.config_dir,
                                       target=self.target,
                                       dependency=conseq_task,
                                       input_path=gt_path[0],
                                       input_key=gt_path[1],
                                       graph_path=graph_out,
                                       output_key=graph_key,
                                       n_scales=n_scales)

            features_out = os.path.join(tmp_folder, 'features_gt.n5')
            
            membrane_features = RegionFeaturesWorkflow(tmp_folder=tmp_folder,
                       max_jobs=self.max_jobs,
                       config_dir=self.config_dir,
                       target=self.target,
                       dependency=graph_task,
                       input_path=input_path[0][0],
                       input_key=input_path[1][0],
                       labels_path=gt_path[0],
                       labels_key=gt_path[1],
                       output_path=features_out,
                       output_key='features_membranes',
                       feature_list=self.region_features,
                       number_of_labels=-1,
                       blockwise=False)

            nuclei_features = RegionFeaturesWorkflow(tmp_folder=tmp_folder,
                       max_jobs=self.max_jobs,
                       config_dir=self.config_dir,
                       target=self.target,
                       dependency=membrane_features,
                       input_path=input_path[0][1],
                       input_key=input_path[1][1],
                       labels_path=gt_path[0],
                       labels_key=gt_path[1],
                       output_path=features_out,
                       output_key='features_nuclei',
                       feature_list=self.region_features,
                       number_of_labels=-1,
                       blockwise=False)

            probs_features = RegionFeaturesWorkflow(tmp_folder=tmp_folder,
                       max_jobs=self.max_jobs,
                       config_dir=self.config_dir,
                       target=self.target,
                       dependency=nuclei_features,
                       input_path=probs_path[0],
                       input_key=probs_path[1],
                       labels_path=gt_path[0],
                       labels_key=gt_path[1],
                       output_path=features_out,
                       output_key='features_probs',
                       feature_list=self.region_features,
                       number_of_labels=-1,
                       blockwise=False)

            mt = getattr(merge_tasks, self._get_task_name('MergeRegionFeatures'))
            merge_task = mt(tmp_folder=tmp_folder,
                            max_jobs=self.max_jobs,
                            config_dir=self.config_dir,
                            dependency=probs_features,
                            feature_path=features_out,
                            feature_keys=['features_membranes', 'features_nuclei', 'features_probs'],
                            feature_list=self.region_features,
                            output_key='features')


            double_edges_out = os.path.join(tmp_folder, 'edge_labels.n5')
            lt = getattr(label_tasks_de,
                         self._get_task_name('NodeSemanticLabels'))
            label_task = lt(tmp_folder=tmp_folder,
                            max_jobs=self.max_jobs,
                            config_dir=self.config_dir,
                            dependency=merge_task,
                            graph_path=graph_out,
                            graph_key=graph_key,
                            semantic_node_labels_path=gt_sem_path[0],
                            semantic_node_labels_key=gt_sem_path[1],
                            semantic_label_fc=1,
                            semantic_label_bg=3,
                            gt_node_labels_path=gt_path[0],
                            gt_node_labels_key=gt_path[1],
                            output_path=double_edges_out,
                            output_key='semantic_label')


            yield label_task
            

class LearningWorkflow(WorkflowBase):
    input_dict = luigi.DictParameter()
    probs_dict = luigi.DictParameter()
    labels_dict = luigi.DictParameter()
    groundtruth_dict = luigi.DictParameter()
    groundtruth_sem_dict = luigi.DictParameter()
    output_path_mc = luigi.Parameter()
    output_path_de = luigi.Parameter()

    region_features = luigi.ListParameter()
    de_region_features = luigi.ListParameter()

    def _check_input(self):
        assert self.input_dict.keys() == self.labels_dict.keys()
        assert self.input_dict.keys() == self.probs_dict.keys()
        assert self.input_dict.keys() == self.groundtruth_dict.keys()
        assert self.input_dict.keys() == self.groundtruth_sem_dict.keys()

    def requires(self):
        self._check_input()
        n_scales = 1

        edge_labels_dict_mc = {}
        edge_labels_dict_de = {}
        features_dict_mc = {}
        features_dict_de = {}

        try:
            os.mkdir(self.tmp_folder)
        except OSError:
            pass

        task_list = list()
        for key, _ in self.input_dict.items():
            tmp_folder = os.path.join(self.tmp_folder, key)
            features_dict_mc[key] = (os.path.join(tmp_folder, 'features.n5'), 'features')
            features_dict_de[key] = (os.path.join(tmp_folder, 'features_gt.n5'), 'features')
            edge_labels_out = os.path.join(tmp_folder, 'edge_labels.n5')
            edge_labels_dict_mc[key] = (edge_labels_out, 'edge_labels')
            edge_labels_dict_de[key] = (edge_labels_out, 'semantic_label')

        f_task_mc = LearningWorkflowPreparationTaskMC(input_dict=self.input_dict,
                                                      probs_dict=self.probs_dict, 
                                                      labels_dict=self.labels_dict, 
                                                      groundtruth_dict=self.groundtruth_dict,
                                                      region_features=self.region_features,
                                                      tmp_folder=self.tmp_folder,
                                                      dependency=self.dependency,
                                                      max_jobs=self.max_jobs,
                                                      config_dir=self.config_dir,
                                                      target=self.target
                                                      )


        f_task_de = LearningWorkflowPreparationTaskDE(input_dict=self.input_dict,
                                                      probs_dict=self.probs_dict, 
                                                      groundtruth_dict=self.groundtruth_dict,
                                                      groundtruth_sem_dict=self.groundtruth_sem_dict,
                                                      region_features=self.de_region_features,
                                                      tmp_folder=self.tmp_folder,
                                                      dependency=f_task_mc,
                                                      max_jobs=self.max_jobs,
                                                      config_dir=self.config_dir,
                                                      target=self.target
                                                      )
            
        learn_task = getattr(learn_tasks,
                             self._get_task_name('LearnRF'))
        rf_task_mc = learn_task(tmp_folder=os.path.join(self.tmp_folder,"multicut_rf"),
                               config_dir=self.config_dir,
                               max_jobs=self.max_jobs,
                               features_dict=features_dict_mc,
                               labels_dict=edge_labels_dict_mc,
                               output_path=self.output_path_mc,
                               dependency=f_task_de)
        rf_task_de = learn_task(tmp_folder=os.path.join(self.tmp_folder,"double_edge_rf"),
                               config_dir=self.config_dir,
                               max_jobs=self.max_jobs,
                               features_dict=features_dict_de,
                               labels_dict=edge_labels_dict_de,
                               output_path=self.output_path_de,
                               dependency=rf_task_mc)
        return rf_task_de

    @staticmethod
    def get_config():
        configs = {'learn_rf': learn_tasks.LearnRFLocal.default_task_config(),
                   'edge_labels': label_tasks_mc.EdgeLabelsLocal.default_task_config(),
                   **GraphWorkflow.get_config(),
                   **EdgeFeaturesWorkflowWithRegionFeatures.get_config()}
        return configs
