import time
from concurrent import futures
from functools import partial

import numpy as np
import nifty
import nifty.ufd as nufd
import nifty.graph.opt.multicut as nmc
import nifty.graph.opt.lifted_multicut as nlmc
import nifty.graph.agglo as nagglo

from vigra.analysis import relabelConsecutive

try:
    from affogato.segmentation import compute_mws_segmentation
    from affogato.segmentation import MWSGridGraph, compute_mws_clustering
except ImportError:
    compute_mws_segmentation = None


# TODO logging
def multicut_kernighan_lin(graph, costs, warmstart=True, time_limit=None, n_threads=1):
    objective = nmc.multicutObjective(graph, costs)
    solver = objective.kernighanLinFactory(warmStartGreedy=warmstart).create(objective)
    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)


def multicut_gaec(graph, costs, time_limit=None, n_threads=1):
    objective = nmc.multicutObjective(graph, costs)
    solver = objective.greedyAdditiveFactory().create(objective)
    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)


def multicut_decomposition(graph, costs, time_limit=None, n_threads=1,
                           solver='kernighan-lin'):

    # get the agglomerator
    agglomerator = key_to_agglomerator(solver)

    # merge attractive edges with ufd to
    # obtain natural connected components
    merge_edges = costs > 0
    ufd = nufd.ufd(graph.numberOfNodes)
    uv_ids = graph.uvIds()
    ufd.merge(uv_ids[merge_edges])
    cc_labels = ufd.elementLabeling()

    # relabel component ids consecutively
    cc_labels, max_id, _ = relabelConsecutive(cc_labels, start_label=0,
                                              keep_zeros=False)

    # TODO use c++ (Thorsten already has impl ?!)
    # TODO check that relabelConsecutive lifts gil ....
    # solve a component sub-problem
    def solve_component(component_id):

        # extract the nodes in this component
        sub_nodes = np.where(cc_labels == component_id)[0].astype('uint64')
        # if we only have a single node, return trivial labeling
        if len(sub_nodes) == 1:
            return sub_nodes, np.array([0], dtype='uint64'), 1

        # extract the subgraph corresponding to this component
        inner_edges, _ = graph.extractSubgraphFromNodes(sub_nodes)
        sub_uvs = uv_ids[inner_edges]
        assert len(inner_edges) == len(sub_uvs), "%i, %i" % (len(inner_edges), len(sub_uvs))

        # relabel sub-nodes and associated uv-ids
        sub_nodes_relabeled, max_local, node_mapping = relabelConsecutive(sub_nodes,
                                                                          start_label=0,
                                                                          keep_zeros=False)
        sub_uvs = nifty.tools.takeDict(node_mapping, sub_uvs)

        # build the graph
        sub_graph = nifty.graph.undirectedGraph(max_local + 1)
        sub_graph.insertEdges(sub_uvs)

        # solve local multicut
        sub_costs = costs[inner_edges]
        assert len(sub_costs) == sub_graph.numberOfEdges, "%i, %i" % (len(sub_costs),
                                                                      sub_graph.numberOfEdges)
        sub_labels = agglomerator(sub_graph, sub_costs, time_limit=time_limit)
        # relabel the solution
        sub_labels, max_seg_local, _ = relabelConsecutive(sub_labels, start_label=0,
                                                          keep_zeros=False)
        assert len(sub_labels) == len(sub_nodes), "%i, %i" % (len(sub_labels), len(sub_nodes))
        return sub_nodes, sub_labels, max_seg_local + 1

    # solve all components in parallel
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(solve_component, component_id)
                 for component_id in range(max_id + 1)]
        results = [t.result() for t in tasks]

    sub_nodes = [res[0] for res in results]
    sub_results = [res[1] for res in results]
    offsets = np.array([res[2] for res in results], dtype='uint64')

    # make proper offsets for the component results
    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)

    # insert sub-results into the components
    node_labels = np.zeros_like(cc_labels, dtype='uint64')

    def insert_solution(component_id):
        nodes = sub_nodes[component_id]
        node_labels[nodes] = (sub_results[component_id] + offsets[component_id])

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(insert_solution, component_id)
                 for component_id in range(max_id + 1)]
        [t.result() for t in tasks]

    return node_labels


# TODO
def multicut_fusion_moves(graph, costs, time_limit=None, n_threads=1,
                          solver='kernighan-lin'):
    assert solver in ('kernighan-lin', 'greedy-additive')
    objective = nmc.multicutObjective(graph, costs)
    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)


def key_to_agglomerator(key):
    agglo_dict = {'kernighan-lin': multicut_kernighan_lin,
                  'greedy-additive': multicut_gaec,
                  'decomposition': multicut_decomposition,
                  'decomposition-gaec': partial(multicut_decomposition,
                                                solver='greedy-additive'),
                  'fusion-moves': multicut_fusion_moves}
    assert key in agglo_dict, key
    return agglo_dict[key]


def lifted_multicut_kernighan_lin(graph, costs, lifted_uv_ids, lifted_costs,
                                  warmstart=True, time_limit=None, n_threads=1):
    objective = nlmc.liftedMulticutObjective(graph)
    objective.setGraphEdgesCosts(costs)
    objective.setCosts(lifted_uv_ids, lifted_costs)
    solver_kl = objective.liftedMulticutKernighanLinFactory().create(objective)
    if time_limit is None:
        if warmstart:
            solver_gaec = objective.liftedMulticutGreedyAdditiveFactory().create(objective)
            res = solver_gaec.optimize()
            return solver_kl.optimize(nodeLabels=res)
        else:
            return solver_kl.optimize()
    else:
        if warmstart:
            solver_gaec = objective.liftedMulticutGreedyAdditiveFactory().create(objective)
            visitor1 = objective.verboseVisitor(visitNth=1000000,
                                                timeLimitTotal=time_limit)
            t0 = time.time()
            res = solver_gaec.optimize(visitor=visitor1)
            t0 = time.time() - t0
            # time limit is not hard, so t0 might actually be bigger than
            # our time limit already
            if t0 > time_limit:
                return res
            visitor2 = objective.verboseVisitor(visitNth=1000000,
                                                timeLimitTotal=time_limit - t0)
            return solver_kl.optimize(nodeLabels=res,
                                      visitor=visitor2)

        else:
            visitor = objective.verboseVisitor(visitNth=1000000,
                                               timeLimitTotal=time_limit)
            return solver_kl.optimize(visitor=visitor)


def lifted_multicut_gaec(graph, costs, lifted_uv_ids, lifted_costs,
                         time_limit=None, n_threads=1):
    objective = nlmc.liftedMulticutObjective(graph)
    objective.setGraphEdgesCosts(costs)
    objective.setCosts(lifted_uv_ids, lifted_costs)
    solver = objective.liftedMulticutGreedyAdditiveFactory().create(objective)
    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)


# TODO
def lifted_multicut_fusion_moves(graph, costs, lifted_uv_ids, lifted_costs,
                                 time_limit=None, n_threads=1, solver='kernighan-lin'):
    assert solver in ('kernighan-lin', 'greedy-additive')
    objective = nlmc.liftedMulticutObjective(graph)
    objective.setGraphEdgesCosts(costs)
    objective.setCosts(lifted_uv_ids, lifted_costs)
    if time_limit is None:
        return solver.optimize()
    else:
        visitor = objective.verboseVisitor(visitNth=1000000,
                                           timeLimitTotal=time_limit)
        return solver.optimize(visitor=visitor)


def key_to_lifted_agglomerator(key):
    agglo_dict = {'kernighan-lin': lifted_multicut_kernighan_lin,
                  'greedy-additive': lifted_multicut_gaec,
                  'fusion-moves': lifted_multicut_fusion_moves}
    assert key in agglo_dict, key
    return agglo_dict[key]


def mutex_watershed(affs, offsets, strides,
                    randomize_strides=False, mask=None,
                    noise_level=0):
    assert compute_mws_segmentation is not None, "Need affogato for mutex watershed"
    ndim = len(offsets[0])
    if noise_level > 0:
        affs += noise_level * np.random.rand(*affs.shape)
    affs[:ndim] *= -1
    affs[:ndim] += 1
    seg = compute_mws_segmentation(affs, offsets,
                                   number_of_attractive_channels=ndim,
                                   strides=strides, mask=mask,
                                   randomize_strides=randomize_strides)
    relabelConsecutive(seg, out=seg, start_label=1, keep_zeros=mask is not None)
    return seg


def compute_grid_graph(shape, mask=None, seeds=None):
    grid_graph = MWSGridGraph(shape)
    if mask is not None:
        grid_graph.set_mask(mask)
    if seeds is not None:
        grid_graph.set_seeds(seeds)
    return grid_graph


def mutex_watershed_with_seeds(affs, offsets, seeds, strides,
                               randomize_strides=False, mask=None,
                               noise_level=0, return_graph=False,
                               seed_state=None):
    assert compute_mws_segmentation is not None, "Need affogato for mutex watershed"
    ndim = len(offsets[0])
    if noise_level > 0:
        affs += noise_level * np.random.rand(*affs.shape)
    affs[:ndim] *= -1
    affs[:ndim] += 1

    # compute grid graph with seeds and optional mask
    shape = affs.shape[1:]
    grid_graph = compute_grid_graph(shape, mask, seeds)

    # compute nn and mutex nh
    grid_graph.intra_seed_weight = 1  # set intra-seed weight to maximal attractive
    if seed_state is not None:
        attractive_edges, attractive_weights = seed_state['attractive']
        grid_graph.set_seed_state(attractive_edges, attractive_weights)
    uvs, weights = grid_graph.compute_nh_and_weights(np.require(affs[:ndim], requirements='C'),
                                                     offsets[:ndim])

    grid_graph.intra_seed_weight = 0  # set intral-seed weight to minimal repulsive
    if seed_state is not None:
        repulsive_edges, repulsive_weights = seed_state['repulsive']
        grid_graph.clear_seed_state()
        grid_graph.set_seed_state(repulsive_edges, repulsive_weights)
    mutex_uvs, mutex_weights = grid_graph.compute_nh_and_weights(np.require(affs[ndim:],
                                                                            requirements='C'),
                                                                 offsets[ndim:], strides, randomize_strides)

    # compute the segmentation
    n_nodes = grid_graph.n_nodes
    seg = compute_mws_clustering(n_nodes, uvs, mutex_uvs, weights, mutex_weights)
    relabelConsecutive(seg, out=seg, start_label=1, keep_zeros=mask is not None)
    seg = seg.reshape(shape)
    if mask is not None:
        seg[np.logical_not(mask)] = 0

    if return_graph:
        return seg, grid_graph
    else:
        return seg


def mala_clustering(graph, edge_features, edge_sizes, threshold):
    n_nodes = graph.numberOfNodes
    policy = nagglo.malaClusterPolicy(graph=graph,
                                      edgeIndicators=edge_features,
                                      nodeSizes=np.zeros(n_nodes, dtype='float'),
                                      edgeSizes=edge_sizes,
                                      threshold=threshold)
    clustering = nagglo.agglomerativeClustering(policy)
    clustering.run()
    return clustering.result()


def agglomerative_clustering(graph, edge_features,
                             node_sizes, edge_sizes,
                             n_stop, size_regularizer):
    policy = nagglo.edgeWeightedClusterPolicy(graph=graph,
                                              edgeIndicators=edge_features,
                                              nodeSizes=node_sizes.astype('float'),
                                              edgeSizes=edge_sizes.astype('float'),
                                              numberOfNodesStop=n_stop,
                                              sizeRegularizer=size_regularizer)
    clustering = nagglo.agglomerativeClustering(policy)
    clustering.run()
    return clustering.result()
