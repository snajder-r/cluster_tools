import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from . import predict as predict_tasks
from . import probs_to_costs as transform_tasks


class EdgeCostsWorkflow(WorkflowBase):

    features_path = luigi.Parameter()
    features_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    node_label_dict = luigi.DictParameter(default={})
    rf_path = luigi.Parameter(default='')
    edge_classes = luigi.ListParameter(default=[1])

    def _costs_with_rf(self):
        print("PREDICTING USING RF")
        predict_task = getattr(predict_tasks,
                               self._get_task_name('Predict'))
        t1 = predict_task(tmp_folder=self.tmp_folder,
                          max_jobs=self.max_jobs,
                          config_dir=self.config_dir,
                          rf_path=self.rf_path,
                          edge_classes=self.edge_classes,
                          features_path=self.features_path,
                          features_key=self.features_key,
                          output_path=self.output_path,
                          output_key="proba_pred",
                          output_labels_key="label_pred",
                          dependency=self.dependency)
        transform_task = getattr(transform_tasks,
                                 self._get_task_name('ProbsToCosts'))
        t2 = transform_task(tmp_folder=self.tmp_folder,
                            max_jobs=self.max_jobs,
                            config_dir=self.config_dir,
                            input_path=self.output_path,
                            input_key="proba_pred",
                            features_path=self.features_path,
                            features_key=self.features_key,
                            output_path=self.output_path,
                            output_key=self.output_key,
                            dependency=t1,
                            node_label_dict=self.node_label_dict)
        return t2

    def _costs(self):
        print("NO RF :-(")
        transform_task = getattr(transform_tasks,
                                 self._get_task_name('ProbsToCosts'))
        t1 = transform_task(tmp_folder=self.tmp_folder,
                            max_jobs=self.max_jobs,
                            config_dir=self.config_dir,
                            input_path=self.features_path,
                            input_key=self.features_key,
                            features_path=self.features_path,
                            features_key=self.features_key,
                            output_path=self.output_path,
                            output_key=self.output_key,
                            dependency=self.dependency,
                            node_label_dict=self.node_label_dict)
        return t1

    def requires(self):
        if self.rf_path == '':
            return self._costs()
        else:
            return self._costs_with_rf()

    @staticmethod
    def get_config():
        configs = super(EdgeCostsWorkflow, EdgeCostsWorkflow).get_config()
        configs.update({'probs_to_costs':
                        transform_tasks.ProbsToCostsLocal.default_task_config()})
        return configs
