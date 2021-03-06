import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from ..utils import volume_utils as vu
from . import skeletonize as skeleton_tasks
from . import upsample_skeletons as upsample_tasks


class SkeletonWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_prefix = luigi.Parameter()
    output_path = luigi.Parameter()
    output_prefix = luigi.Parameter()
    work_scale = luigi.IntParameter()
    target_scale = luigi.IntParameter(default=None)
    skeleton_format = luigi.Parameter(default='n5')

    def _read_n_labels(self, in_key):
        with vu.file_reader(self.input_path, 'r') as f:
            try:
                max_id = f[in_key].attrs['maxId']
            except KeyError:
                try:
                    max_id = f[self.input_prefix].attrs['maxId']
                except KeyError:
                    raise KeyError("Could not find maxId attribute in %s in keys %s or %s" % (self.input_path,
                                                                                              in_key, self.input_prefix))
        return int(max_id) + 1

    def requires(self):
        skel_task = getattr(skeleton_tasks,
                            self._get_task_name('Skeletonize'))
        in_key1 = '%s/s%i' % (self.input_prefix, self.work_scale)
        out_key1 = '%s/s%i' % (self.output_prefix, self.work_scale)

        # read the number of labels
        n_labels = self._read_n_labels(in_key1)

        dep = skel_task(tmp_folder=self.tmp_folder,
                        max_jobs=self.max_jobs,
                        config_dir=self.config_dir,
                        dependency=self.dependency,
                        input_path=self.input_path,
                        input_key=in_key1,
                        output_path=self.output_path,
                        output_key=out_key1,
                        number_of_labels=n_labels,
                        skeleton_format=self.skeleton_format)

        # check if we have a target scale to upsample skeletons to
        target_scale = self.work_scale if self.target_scale is None else self.target_scale
        if target_scale == self.work_scale:
            return dep
        else:
            assert target_scale < self.work_scale, "%i, %i" % (target_scale, self.work_scale)
            raise NotImplementedError("Skeleton upsampling not implemented yet")
        upsample_task = getattr(upsample_tasks,
                                self._get_task_name('UpsampleSkeletons'))
        in_key2 = '%s/s%i' % (self.input_prefix, self.target_scale)
        out_key2 = '%s/s%i' % (self.output_prefix, self.target_scale)
        dep = upsample_task(tmp_folder=self.tmp_folder,
                            max_jobs=self.max_jobs,
                            config_dir=self.config_dir,
                            dependency=dep,
                            input_path=self.input_path,
                            input_key=in_key2,
                            skeleton_path=self.output_path,
                            skeleton_key=out_key1,
                            output_path=self.output_path,
                            output_key=out_key2)
        return t2

    @staticmethod
    def get_config():
        configs = super(SkeletonWorkflow, SkeletonWorkflow).get_config()
        configs.update({'skeletonize': skeleton_tasks.SkeletonizeLocal.default_task_config(),
                        'upsample_skeletons': upsample_tasks.UpsampleSkeletonsLocal.default_task_config()})
        return configs
