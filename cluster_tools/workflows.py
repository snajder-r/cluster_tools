import os
import luigi

from .cluster_tasks import WorkflowBase
from .watershed import WatershedWorkflow
from .graph import GraphWorkflow

# TODO more features and options to choose which features to choose
from .features import EdgeFeaturesWorkflow, EdgeFeaturesWorkflowWithRegionFeatures
from .features import RegionFeaturesWorkflow
from .costs import EdgeCostsWorkflow

# TODO more options for lifted problems
from .lifted_features import LiftedFeaturesFromNodeLabelsWorkflow

from .multicut import MulticutWorkflow
from .region_to_edge_features import R2EFeaturesWorkflow
from .lifted_multicut import LiftedMulticutWorkflow

from .debugging import CheckSubGraphsWorkflow
from . import write as write_tasks

#
from .agglomerative_clustering import agglomerative_clustering as agglomerate_tasks
from .stitching import StitchingAssignmentsWorkflow


from .classifyrf import ClassificationWorkflow

# TODO add options to choose which features to use
# NOTE in the current implementation, we can only compute the
# graph with n_scales=1, otherwise we will clash with the multicut merged graphs
class ProblemWorkflow(WorkflowBase):
    img_path = luigi.Parameter()
    img_key = luigi.Parameter()
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    ws_path = luigi.Parameter()
    ws_key = luigi.Parameter()
    problem_path = luigi.Parameter()

    # optional params for costs
    rf_path = luigi.Parameter(default='')
    edge_classes = luigi.ListParameter(default=[1])
    node_label_dict = luigi.DictParameter(default={})
    region_features = luigi.ListParameter()
    
    max_jobs_merge = luigi.IntParameter(default=1)
    # do we compte costs
    compute_costs = luigi.BoolParameter(default=True)
    # do we run sanity checks ?
    sanity_checks = luigi.BoolParameter(default=False)
    
    

    # hard-coded keys
    graph_key = 's0/graph'
    features_key = 'features'

    costs_key = 's0/costs'

    def requires(self):
        dep = GraphWorkflow(tmp_folder=self.tmp_folder,
                            max_jobs=self.max_jobs,
                            config_dir=self.config_dir,
                            target=self.target,
                            dependency=self.dependency,
                            input_path=self.ws_path,
                            input_key=self.ws_key,
                            graph_path=self.problem_path,
                            output_key=self.graph_key,
                            n_scales=1)
        # sanity check the subgraph
        if self.sanity_checks:
            graph_block_prefix = os.path.join(self.problem_path,
                                              's0', 'sub_graphs', 'block_')
            dep = CheckSubGraphsWorkflow(tmp_folder=self.tmp_folder,
                                         max_jobs=self.max_jobs,
                                         config_dir=self.config_dir,
                                         target=self.target,
                                         ws_path=self.ws_path,
                                         ws_key=self.ws_key,
                                         graph_block_prefix=graph_block_prefix,
                                         dependency=dep)
        dep = EdgeFeaturesWorkflowWithRegionFeatures(tmp_folder=self.tmp_folder,
                                             max_jobs=self.max_jobs,
                                             config_dir=self.config_dir,
                                             dependency=dep,
                                             target=self.target,
                                             input_path=self.img_path,
                                             input_key=self.img_key,
                                             probs_path=self.input_path,
                                             probs_key=self.input_key,
                                             labels_path=self.ws_path,
                                             labels_key=self.ws_key,
                                             graph_path=self.problem_path,
                                             graph_key=self.graph_key,
                                             output_path=self.problem_path,
                                             output_key=self.features_key,
                                             region_features=self.region_features)
        if self.compute_costs:
            dep = EdgeCostsWorkflow(tmp_folder=self.tmp_folder,
                                    max_jobs=self.max_jobs,
                                    config_dir=self.config_dir,
                                    target=self.target,
                                    dependency=dep,
                                    features_path=self.problem_path,
                                    features_key=self.features_key,
                                    output_path=self.problem_path,
                                    output_key=self.costs_key,
                                    node_label_dict=self.node_label_dict,
                                    rf_path=self.rf_path, 
                                    edge_classes=self.edge_classes)
        return dep

    @staticmethod
    def get_config():
        config = {**GraphWorkflow.get_config(),
                  **EdgeFeaturesWorkflow.get_config(),
                  **RegionFeaturesWorkflow.get_config(),
                  **EdgeCostsWorkflow.get_config()}
        return config


class SegmentationWorkflowBase(WorkflowBase):
    img_path = luigi.Parameter()
    img_key = luigi.Parameter()
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    # where to save the watersheds
    ws_path = luigi.Parameter()
    ws_key = luigi.Parameter()
    # where to save the problem (graph, edge_features etc)
    problem_path = luigi.Parameter()
    # where to save the node labels
    node_labels_key = luigi.Parameter()
    # where to save the resulting segmentation
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    # optional path to mask
    mask_path = luigi.Parameter(default='')
    mask_key = luigi.Parameter(default='')

    # optional path for random forest used for cost computation
    rf_path = luigi.Parameter(default='')
    edge_classes = luigi.ListParameter(default=[1])
    # node label dict: dictionary for additional node labels used in costs
    node_label_dict = luigi.DictParameter(default={})

    # number of jobs used for merge tasks
    max_jobs_merge = luigi.IntParameter(default=1)
    # skip watershed (watershed volume must already be preset)
    skip_ws = luigi.BoolParameter(default=False)
    # run agglomeration immediately after watersheds
    agglomerate_ws = luigi.BoolParameter(default=False)
    # run some sanity checks for intermediate results
    sanity_checks = luigi.BoolParameter(default=False)
    region_features = luigi.ListParameter()

    # hard-coded keys
    graph_key = 's0/graph'
    features_key = 'features'
    costs_key = 's0/costs'

    def _watershed_tasks(self):
        if self.skip_ws:
            assert os.path.exists(os.path.join(self.ws_path, self.ws_key)), "%s:%s" % (self.ws_path,
                                                                                       self.ws_key)
            return self.dependency
        else:
            dep = WatershedWorkflow(tmp_folder=self.tmp_folder,
                                    max_jobs=self.max_jobs,
                                    config_dir=self.config_dir,
                                    target=self.target,
                                    dependency=self.dependency,
                                    input_path=self.input_path,
                                    input_key=self.input_key,
                                    output_path=self.ws_path,
                                    output_key=self.ws_key,
                                    mask_path=self.mask_path,
                                    mask_key=self.mask_key,
                                    agglomeration=self.agglomerate_ws)
            return dep

    def _problem_tasks(self, dep, compute_costs):
        dep = ProblemWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                              max_jobs=self.max_jobs, target=self.target, dependency=dep,
                              input_path=self.input_path, input_key=self.input_key,
                              img_path=self.img_path, img_key=self.img_key,
                              ws_path=self.ws_path, ws_key=self.ws_key,
                              problem_path=self.problem_path, rf_path=self.rf_path,
                              edge_classes=self.edge_classes,
                              node_label_dict=self.node_label_dict,
                              max_jobs_merge=self.max_jobs_merge,
                              compute_costs=compute_costs, sanity_checks=self.sanity_checks,
                              region_features=self.region_features)
        return dep

    def _write_tasks(self, dep, identifier):
        if self.output_key == '':
            return dep
        write_task = getattr(write_tasks, self._get_task_name('Write'))
        dep = write_task(tmp_folder=self.tmp_folder,
                         max_jobs=self.max_jobs,
                         config_dir=self.config_dir,
                         dependency=dep,
                         input_path=self.ws_path,
                         input_key=self.ws_key,
                         output_path=self.output_path,
                         output_key=self.output_key,
                         assignment_path=self.output_path,
                         assignment_key=self.node_labels_key,
                         identifier=identifier)
        return dep

    @staticmethod
    def get_config():
        config = {**WatershedWorkflow.get_config(), **ProblemWorkflow.get_config()}
        return config


class MulticutSegmentationWorkflow(SegmentationWorkflowBase):
    # number of jobs used for sub multicuts
    max_jobs_multicut = luigi.IntParameter(default=1)
    # number of scales
    n_scales = luigi.IntParameter()

    def _multicut_tasks(self, dep):
        dep = MulticutWorkflow(tmp_folder=self.tmp_folder,
                               max_jobs=self.max_jobs_multicut,
                               config_dir=self.config_dir,
                               target=self.target,
                               dependency=dep,
                               problem_path=self.problem_path,
                               n_scales=self.n_scales,
                               assignment_path=self.output_path,
                               assignment_key=self.node_labels_key)
        return dep

    def requires(self):
        dep = self._watershed_tasks()
        dep = self._problem_tasks(dep, compute_costs=True)
        dep = self._multicut_tasks(dep)
        dep = self._write_tasks(dep, 'multicut')
        return dep

    @staticmethod
    def get_config():
        config = super(MulticutSegmentationWorkflow, MulticutSegmentationWorkflow).get_config()
        config.update(MulticutWorkflow.get_config())
        return config

class MulticutSegmentationWithDoubleEdgesWorkflow(SegmentationWorkflowBase):
    # number of jobs used for sub multicuts
    max_jobs_multicut = luigi.IntParameter(default=1)
    # number of scales
    n_scales = luigi.IntParameter()
    double_edge_path = luigi.Parameter()

    
    def _double_edge_prediction_task(self,dep):
        dep = ClassificationWorkflow(tmp_folder=self.tmp_folder,
                               max_jobs=self.max_jobs_multicut,
                               config_dir=self.config_dir,
                               target=self.target,
                               rf_path=self.rf_path, 
                               features_path=self.problem_path,
                               features_key=self.features_key,
                               output_path=self.double_edge_path,
                               output_key='labels',
                               dependency=dep)
        return dep
    
    def _multicut_tasks(self, dep):
        dep = MulticutWorkflow(tmp_folder=self.tmp_folder,
                               max_jobs=self.max_jobs_multicut,
                               config_dir=self.config_dir,
                               target=self.target,
                               dependency=dep,
                               problem_path=self.problem_path,
                               n_scales=self.n_scales,
                               assignment_path=self.output_path,
                               assignment_key=self.node_labels_key)
        return dep
    
    def requires(self):
        dep = self._watershed_tasks()
        dep = self._problem_tasks(dep, compute_costs=True)
        dep = self._multicut_tasks(dep)
        dep = self._write_tasks(dep, 'multicut')
        dep = self._double_edge_prediction_task(dep)
        
        return dep

    
    @staticmethod
    def get_config():
        config = super(MulticutSegmentationWithDoubleEdgesWorkflow, MulticutSegmentationWithDoubleEdgesWorkflow).get_config()
        config.update(MulticutWorkflow.get_config())
        return config
        

class LiftedMulticutSegmentationWorkflow(SegmentationWorkflowBase):
    # number of jobs used for sub multicuts
    max_jobs_multicut = luigi.IntParameter(default=1)
    # number of scales
    n_scales = luigi.IntParameter()

    # node labels for lifted milticut
    lifted_labels_path = luigi.Parameter()
    lifted_labels_key = luigi.Parameter()
    lifted_prefix = luigi.Parameter()
    # graph depth for lifted neighborhood
    nh_graph_depth = luigi.IntParameter(default=4)

    # TODO different options for lifted problems
    def _lifted_problem_tasks(self, dep):
        nh_key = 's0/lifted_nh_%s' % self.lifted_prefix
        feat_key = 's0/lifted_costs_%s' % self.lifted_prefix
        dep = LiftedFeaturesFromNodeLabelsWorkflow(tmp_folder=self.tmp_folder,
                                                   max_jobs=self.max_jobs,
                                                   config_dir=self.config_dir,
                                                   target=self.target,
                                                   dependency=dep,
                                                   ws_path=self.ws_path,
                                                   ws_key=self.ws_key,
                                                   labels_path=self.lifted_labels_path,
                                                   labels_key=self.lifted_labels_key,
                                                   output_path=self.problem_path,
                                                   nh_out_key=nh_key,
                                                   feat_out_key=feat_key,
                                                   graph_path=self.problem_path,
                                                   graph_key=self.graph_key,
                                                   prefix=self.lifted_prefix,
                                                   nh_graph_depth=self.nh_graph_depth)
        return dep

    def _lifted_multicut_tasks(self, dep):
        dep = LiftedMulticutWorkflow(tmp_folder=self.tmp_folder,
                                     max_jobs=self.max_jobs_multicut,
                                     config_dir=self.config_dir,
                                     target=self.target,
                                     dependency=dep,
                                     problem_path=self.problem_path,
                                     n_scales=self.n_scales,
                                     assignment_path=self.output_path,
                                     assignment_key=self.node_labels_key,
                                     lifted_prefix=self.lifted_prefix)
        return dep

    def requires(self):
        dep = self._watershed_tasks()
        dep = self._problem_tasks(dep, compute_costs=True)
        dep = self._lifted_problem_tasks(dep)
        dep = self._lifted_multicut_tasks(dep)
        dep = self._write_tasks(dep, 'lifted_multicut')
        return dep

    @staticmethod
    def get_config():
        config = super(LiftedMulticutSegmentationWorkflow,
                       LiftedMulticutSegmentationWorkflow).get_config()
        config.update({**LiftedFeaturesFromNodeLabelsWorkflow.get_config(),
                       **LiftedMulticutWorkflow.get_config()})
        return config


# TODO support vanilla agglomerative clustering
class AgglomerativeClusteringWorkflow(SegmentationWorkflowBase):
    threshold = luigi.FloatParameter()

    def _agglomerate_task(self, dep):
        agglomerate_task = getattr(agglomerate_tasks,
                                   self._get_task_name('AgglomerativeClustering'))
        dep = agglomerate_task(tmp_folder=self.tmp_folder,
                               max_jobs=self.max_jobs,
                               config_dir=self.config_dir,
                               problem_path=self.problem_path,
                               assignment_path=self.output_path,
                               assignment_key=self.node_labels_key,
                               features_path=self.problem_path,
                               features_key=self.features_key,
                               threshold=self.threshold,
                               dependency=dep)
        return dep

    def requires(self):
        dep = self._watershed_tasks()
        dep = self._problem_tasks(dep, compute_costs=False)
        dep = self._agglomerate_task(dep)
        dep = self._write_tasks(dep, 'agglomerative_clustering')
        return dep

    @staticmethod
    def get_config():
        configs = super(AgglomerativeClusteringWorkflow,
                        AgglomerativeClusteringWorkflow).get_config()
        configs.update({'agglomerative_clustering':
                        agglomerate_tasks.AgglomerativeClusteringLocal.default_task_config()})
        return configs


class SimpleStitchingWorkflow(MulticutSegmentationWorkflow):
    edge_size_threshold = luigi.IntParameter()

    def _simple_stitcher(self, dep):
        dep = StitchingAssignmentsWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                                           max_job=self.max_jobs, target=self.target, dependency=dep,
                                           problem_path=self.problem_path,
                                           labels_path=self.ws_path, labels_key=self.ws_key,
                                           assignment_path=self.output_path, assignments_key=self.node_labels_key,
                                           features_key=self.features_key, graph_key=self.graph_key,
                                           edge_size_threshold=self.edge_size_threshold)
        return dep

    def requires(self):
        dep = self._watershed_tasks()
        dep = self._problem_tasks(dep, compute_costs=False)
        dep = self._simple_stitcher(dep)
        dep = self._write_tasks(dep, 'simple_stitching')
        return dep

    @staticmethod
    def get_config():
        configs = super(SimpleStitchingWorkflow,
                        SimpleStitchingWorkflow).get_config()
        configs.update(StitchingAssignmentsWorkflow.get_config())
        return configs
