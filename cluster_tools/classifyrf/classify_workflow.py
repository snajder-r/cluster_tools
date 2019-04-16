import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from . import classifyrf as predict_tasks

class ClassificationWorkflow(WorkflowBase):

    # input and output volumes
    rf_path = luigi.Parameter()
    features_path = luigi.Parameter()
    features_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()

    def _classify_with_rf(self):
        predict_task = getattr(predict_tasks,
                               self._get_task_name('ClassifyRF'))
                               
        t = predict_task(tmp_folder=self.tmp_folder,
                         max_jobs=self.max_jobs,
                         config_dir=self.config_dir,
                         rf_path=self.rf_path,
                         features_path=self.features_path,
                         features_key=self.features_key,
                         output_path=self.output_path,
                         output_key=self.output_key,
                         dependency=self.dependency)
      
        return t


    def requires(self):
        return self._classify_with_rf()

    @staticmethod
    def get_config():
        configs = super(ClassificationWorkflow, ClassificationWorkflow).get_config()
        configs.update({'ClassifyRF':
                        predict_tasks.ClassifyRFLocal.default_task_config()})
        return configs
