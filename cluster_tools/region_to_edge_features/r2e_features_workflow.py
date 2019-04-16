import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from . import transform_features as transform_tasks


class R2EFeaturesWorkflow(WorkflowBase):
    
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    region_feature_paths = luigi.ListParameter()
    region_feature_keys = luigi.ListParameter()
    edge_feature_path = luigi.Parameter()
    edge_feature_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    

    def _check_input(self):
        pass

    def requires(self):
        self._check_input()
        n_scales = 1

        try:
            os.mkdir(self.tmp_folder)
        except OSError:
            pass
        transform_task = getattr(transform_tasks, self._get_task_name('R2EFeatures'))
        dep = transform_task(tmp_folder=self.tmp_folder,
                             config_dir=self.config_dir,
                             max_jobs=self.max_jobs,
                             graph_path=self.graph_path,
                             graph_key=self.graph_key,
                             region_feature_paths=self.region_feature_paths,
                             region_feature_keys=self.region_feature_keys,
                             edge_feature_path=self.edge_feature_path,
                             edge_feature_key=self.edge_feature_key,
                             output_path=self.output_path,
                             output_key=self.output_key,
                             dependency=self.dependency)
        return dep

    @staticmethod
    def get_config():
        configs = {'r2e_features': transform_tasks.R2EFeaturesLocal.default_task_config()}
        return configs
