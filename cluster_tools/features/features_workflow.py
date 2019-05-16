import os
import json
import luigi

import cluster_tools.utils.volume_utils as vu

from ..cluster_tasks import WorkflowBase
from . import block_edge_features as feat_tasks
from . import merge_edge_features as merge_tasks
from . import region_features as reg_tasks
from . import merge_region_features as merge_reg_tasks
from ..region_to_edge_features import R2EFeaturesWorkflow


class EdgeFeaturesWorkflow(WorkflowBase):

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    labels_path = luigi.Parameter()
    labels_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    blocks_prefix = luigi.Parameter(default='blocks')
    max_jobs_merge = luigi.IntParameter(default=1)

    # for now we only support n5 / zarr input labels
    @staticmethod
    def _check_input(path):
        ending = path.split('.')[-1]
        assert ending.lower() in ('zr', 'zarr', 'n5'),\
            "Only support n5 and zarr files, not %s" % ending

    def requires(self):
        self._check_input(self.input_path)
        self._check_input(self.labels_path)

        feat_task = getattr(feat_tasks,
                            self._get_task_name('BlockEdgeFeatures'))
        dep = feat_task(tmp_folder=self.tmp_folder,
                        max_jobs=self.max_jobs,
                        config_dir=self.config_dir,
                        input_path=self.input_path,
                        input_key=self.input_key,
                        labels_path=self.labels_path,
                        labels_key=self.labels_key,
                        graph_path=self.graph_path,
                        output_path=self.output_path,
                        blocks_prefix=self.blocks_prefix,
                        dependency=self.dependency)
        merge_task = getattr(merge_tasks,
                     self._get_task_name('MergeEdgeFeatures'))
        dep = merge_task(tmp_folder=self.tmp_folder,
                         max_jobs=self.max_jobs_merge,
                         config_dir=self.config_dir,
                         graph_path=self.graph_path,
                         graph_key=self.graph_key,
                         output_path=self.output_path,
                         output_key=self.output_key,
                         blocks_prefix=self.blocks_prefix,
                         dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(EdgeFeaturesWorkflow, EdgeFeaturesWorkflow).get_config()
        configs.update({'block_edge_features': feat_tasks.BlockEdgeFeaturesLocal.default_task_config(),
                        'merge_edge_features': merge_tasks.MergeEdgeFeaturesLocal.default_task_config()})
        return configs
        

# TODO support more than mean value
# TODO support multi-channel inputs
class RegionFeaturesWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    labels_path = luigi.Parameter()
    labels_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    number_of_labels = luigi.Parameter()
    feature_list = luigi.ListParameter()
    blockwise = luigi.ListParameter(True)
    def requires(self):
        # If we wanna do merging, we need to also have the count feature 
        # in the last position
        #if not 'count' in self.feature_list:
        #    self.feature_list = [f for f in self.feature_list if not (f=='count' or f=='Count')]
        #    self.feature_list.append('count')

        region_features_tmp_path = os.path.join(self.tmp_folder, 'region_features_tmp.n5')
        region_features_tmp_key = 'block_feats'
            
        feat_task = getattr(reg_tasks,
                            self._get_task_name('RegionFeatures'))
        dep = feat_task(tmp_folder=self.tmp_folder,
                        max_jobs=self.max_jobs,
                        config_dir=self.config_dir,
                        input_path=self.input_path,
                        input_key=self.input_key,
                        labels_path=self.labels_path,
                        labels_key=self.labels_key,
                        feature_list=self.feature_list,
                        output_path=region_features_tmp_path,
                        output_key=region_features_tmp_key,
                        blockwise=self.blockwise,
                        dependency=self.dependency)
        merge_task = getattr(merge_reg_tasks,
                             self._get_task_name('MergeRegionFeatures'))
        n_labels = self.number_of_labels
        dep = merge_task(tmp_folder=self.tmp_folder,
                         max_jobs=self.max_jobs,
                         config_dir=self.config_dir,
                         output_path=self.output_path,
                         output_key=self.output_key,
                         number_of_labels=n_labels,
                         block_features_path=region_features_tmp_path,
                         block_features_key=region_features_tmp_key,
                         feature_list=self.feature_list,
                         dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(RegionFeaturesWorkflow, RegionFeaturesWorkflow).get_config()
        configs.update({'region_features': reg_tasks.RegionFeaturesLocal.default_task_config(),
                        'merge_region_features': merge_reg_tasks.MergeRegionFeaturesLocal.default_task_config()})
        return configs

        
class EdgeFeaturesWorkflowWithRegionFeatures(WorkflowBase):

    input_paths = luigi.ListParameter()
    input_keys = luigi.ListParameter()
    probs_path = luigi.Parameter()
    probs_key = luigi.Parameter()
    labels_path = luigi.Parameter()
    labels_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    region_features = luigi.ListParameter()
    number_of_labels = luigi.Parameter(default=-1)
    max_jobs_merge = luigi.IntParameter(default=1)
    
    edge_features_probs_key = 'edge_features_probs'
    edge_features_input_key = 'edge_features_input'
    region_features_probs_key = 'region_features_probs'
    region_features_input_key = 'region_features_input'

    use_edge_features = luigi.BoolParameter(default=True)

    # for now we only support n5 / zarr input labels
    @staticmethod
    def _check_input(path):
        ending = path.split('.')[-1]
        assert ending.lower() in ('zr', 'zarr', 'n5'),\
            "Only support n5 and zarr files, not %s" % ending

    def requires(self):
        self._check_input(self.probs_path)
        self._check_input(self.labels_path)
        self._check_input(self.graph_path)

        dep = self.dependency
        edge_feature_paths = []
        edge_feature_keys= []
        region_feature_paths = []
        region_feature_keys = []

        print(self.use_edge_features,"INPUT PATHS I GOT: ", self.input_paths)

        for i in range(len(self.input_paths)):
            input_path = self.input_paths[i]
            input_key = self.input_keys[i]

            print(self.use_edge_features,"INPUT PATH: ", input_path)
            print(self.use_edge_features,"INPUT KEY: ", input_key)
            if self.use_edge_features:
                outfile = self.output_path
                outkey = self.edge_features_input_key+'_ch'+str(i)
                edge_feature_paths.append(outfile)
                edge_feature_keys.append(outkey)

                dep = EdgeFeaturesWorkflow(tmp_folder=self.tmp_folder,
                        max_jobs=self.max_jobs,
                        config_dir=self.config_dir,
                        target=self.target,
                        dependency=dep,
                        input_path=input_path,
                        input_key=input_key,
                        labels_path=self.labels_path,
                        labels_key=self.labels_key,
                        graph_path=self.graph_path,
                        graph_key=self.graph_key,
                        blocks_prefix='blocks_ch'+str(i),
                        output_path=outfile,
                        output_key=outkey,
                        max_jobs_merge=self.max_jobs_merge)

            outfile = self.output_path
            outkey = self.region_features_input_key+'_ch'+str(i)
            region_feature_paths.append(outfile)
            region_feature_keys.append(outkey)
            dep = RegionFeaturesWorkflow(tmp_folder=self.tmp_folder,
                       max_jobs=self.max_jobs,
                       config_dir=self.config_dir,
                       target=self.target,
                       dependency=dep,
                       input_path=input_path,
                       input_key=input_key,
                       labels_path=self.labels_path,
                       labels_key=self.labels_key,
                       output_path=outfile,
                       output_key=outkey,
                       feature_list=self.region_features,
                       number_of_labels=self.number_of_labels,
                       blockwise=False)

        if self.use_edge_features:
            outfile = self.output_path
            outkey = self.edge_features_probs_key
            edge_feature_paths.append(outfile)
            edge_feature_keys.append(outkey)
            dep = EdgeFeaturesWorkflow(tmp_folder=self.tmp_folder,
                    max_jobs=self.max_jobs,
                    config_dir=self.config_dir,
                    target=self.target,
                    dependency=dep,
                    input_path=self.probs_path,
                    input_key=self.probs_key,
                    labels_path=self.labels_path,
                    labels_key=self.labels_key,
                    graph_path=self.graph_path,
                    graph_key=self.graph_key,
                    output_path=outfile,
                    output_key=outkey,
                    max_jobs_merge=self.max_jobs_merge)
        
        outfile = self.output_path
        outkey = self.region_features_probs_key
        region_feature_paths.append(outfile)
        region_feature_keys.append(outkey)
        dep = RegionFeaturesWorkflow(tmp_folder=self.tmp_folder,
                    max_jobs=self.max_jobs,
                    config_dir=self.config_dir,
                    target=self.target,
                    dependency=dep,
                    input_path=self.probs_path,
                    input_key=self.probs_key,
                    labels_path=self.labels_path,
                    labels_key=self.labels_key,
                    output_path=outfile,
                    output_key=outkey,
                    feature_list=self.region_features,
                    number_of_labels=self.number_of_labels,
                    blockwise=False)
       
        print(self.use_edge_features,"MERGE ", region_feature_paths)
        dep = R2EFeaturesWorkflow(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                                   config_dir=self.config_dir, target=self.target,
                                   dependency=dep, graph_path=self.graph_path,
                                   graph_key=self.graph_key,
                                   region_feature_paths=region_feature_paths, region_feature_keys=region_feature_keys,
                                   edge_feature_paths = edge_feature_paths, edge_feature_keys = edge_feature_keys,
                                   output_path = self.output_path, output_key = self.output_key)
        
        return dep

    @staticmethod
    def get_config():
        configs = {**EdgeFeaturesWorkflow.get_config(),
                   **R2EFeaturesWorkflow.get_config(),
                   **RegionFeaturesWorkflow.get_config()}
        return configs
