import os
import json
import luigi

import cluster_tools.utils.volume_utils as vu
from ..cluster_tasks import WorkflowBase
from . import grow_regions as grow_tasks

class DoubleEdgesWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    de_labels_path = luigi.Parameter()
    de_labels_key = luigi.Parameter()
    boundaries_path = luigi.Parameter()
    boundaries_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()  

    output_path = luigi.Parameter()
    output_key = luigi.Parameter()  
    dependency = luigi.TaskParameter()



    def requires(self):
        try:
            os.mkdir(self.tmp_folder)
        except OSError:
            pass

        grow_task = getattr(grow_tasks, self._get_task_name('GrowRegionsTask'))
        gt = grow_task(tmp_folder=os.path.join(self.tmp_folder,"multicut_rf"),
                       config_dir=self.config_dir, max_jobs=self.max_jobs,
                       input_path = self.input_path, input_key = self.input_key,
                       de_labels_path = self.de_labels_path, de_labels_key = self.de_labels_key,
                       boundaries_path = self.boundaries_path, boundaries_key = self.boundaries_key,
                       graph_path = self.graph_path, graph_key = self.graph_key,
                       output_path=self.output_path, output_key=self.output_key,
                       dependency=self.dependency)
        return gt

    @staticmethod
    def get_config():
        configs = {'grow_regions': GrowRegionsTaskLocal.default_task_config()}
        return configs
