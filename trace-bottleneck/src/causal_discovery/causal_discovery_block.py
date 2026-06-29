# causal discovery
# TODO: add more libraries and models for future ablation studies
import torch
import pandas as pd
import numpy as np
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.cit import chisq

#import jpype
#if not jpype.isJVMStarted():
#    jpype.startJVM("-Djava.class.path=" + ":".join(CLASSPATH))


#import pytetrad.tools.translate as tr
#import edu.cmu.tetrad.search as ts
#import edu.cmu.tetrad.search.test as searchTest
#import edu.cmu.tetrad.search.score as searchScore

from src.plots import maybe_plot_graph

def _discretize_column(col_np, n_bins=5, max_categorical_unique=10):
    """Discretize a single column for causal discovery.

    - If the column has <= max_categorical_unique unique values it is treated
      as already categorical and cast to int directly.
    - Otherwise it is quantile-binned into n_bins equal-frequency buckets so
      that chi-squared / G-squared tests can detect dependencies without the
      information loss caused by naive torch.long truncation.
    """
    n_unique = len(np.unique(col_np[~np.isnan(col_np)]))
    if n_unique <= max_categorical_unique:
        return col_np.astype(int)
    # Quantile-based discretization for continuous columns
    try:
        binned = pd.qcut(col_np, q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        binned = pd.cut(col_np, bins=min(n_bins, n_unique), labels=False,
                        duplicates='drop')
    return np.nan_to_num(binned, nan=0).astype(int)


def process_data_for_causal_discovery(data, label_names, causal_discovery_library):
    if causal_discovery_library=="causallearn":
        if not isinstance(data.c, torch.Tensor):
            data.c = torch.tensor(data.c, dtype=torch.float32)
            data.y = torch.tensor(data.y, dtype=torch.float32)
        concat = torch.cat((data.c, data.y.float()), dim=1)

        # Discretize each column properly: quantile-bin continuous concepts
        # instead of truncating with torch.long (which destroys z-scored signals)
        discretized_cols = []
        for ci in range(concat.shape[1]):
            col_np = concat[:, ci].numpy()
            discretized_cols.append(_discretize_column(col_np))
        processed_data = torch.tensor(np.stack(discretized_cols, axis=1),
                                      dtype=torch.long)

    #elif model_name == "pc":
    #    data = torch.cat((data.c, data.y), dim=1).numpy()
    elif causal_discovery_library=="pytetrad":
        concat = torch.cat((data.c, data.y.float()), dim=1)
        # Discretize before converting to pytetrad format
        discretized_cols = []
        for ci in range(concat.shape[1]):
            col_np = concat[:, ci].numpy()
            discretized_cols.append(_discretize_column(col_np))
        processed_data = pd.DataFrame(np.stack(discretized_cols, axis=1).astype(int))
        processed_data.columns = label_names
        processed_data = tr.pandas_data_to_tetrad(processed_data)
    else:
        raise ValueError(f"Unknown causal discovery library: {causal_discovery_library}")
    return processed_data



def apply_causal_discovery(data,
                           causal_discovery_model,
                           causal_discovery_type,
                           causal_discovery_library,
                           **kwargs):
    model_info = dict()
    if causal_discovery_library=="causallearn":
        algo_function = globals().get(causal_discovery_model)
        model_info["method"] = list(kwargs.values())[0]   
        if causal_discovery_type == "constraint-based":
            g = algo_function(data.numpy(), 0.05, model_info["method"])
            model_info["pvalue"] = 0.05
        elif causal_discovery_type == "score-based":
            g = algo_function(data.numpy(), model_info["method"])
            model_info["score"] = g.score if hasattr(g, 'score') else ""
        print(model_info)
        return g, model_info
    elif causal_discovery_library=="pytetrad":
        if causal_discovery_type == "constraint-based":
            if kwargs.get("ind_test") == "chisq":
                test = searchTest.IndTestChiSquare(data, 0.05)
            elif kwargs.get("ind_test") == "gsq":
                test = searchTest.IndTestGSquare(data, 0.05)
            else:
                raise ValueError(f"Unknown independence test: {kwargs.get('ind_test')}")
            #elif kwargs.get("ind_test") == "gaussianLrt":
            #    test = ts.IndTestConditionalGaussianLrt(data, 0.05)
            #elif kwargs.get("ind_test") == "mvpLrt":
            #    test = ts.IndTestMvpLrt(data, 0.05)
            run_algo = getattr(ts, causal_discovery_model.capitalize())(test)
            model_info["method"] = kwargs.get("ind_test")
            model_info["pvalue"] = 0.05
            g = run_algo.search()
            print(model_info)
        elif causal_discovery_type == "score-based":
            if kwargs.get("score_func") == "local_score_BDeu":
                score = searchScore.BdeuScore(data)
            else:
                raise ValueError(f"Unknown score function: {kwargs.get('score_func')}")
            run_algo = getattr(ts, causal_discovery_model.capitalize())(score)
            model_info["method"] = kwargs.get("score_func") 
            g = run_algo.search()
            model_info["score"] = g.getAllAttributes().get("Score")
            print(model_info)
        return g, model_info
    else:
        raise ValueError(f"Unknown causal discovery model: {causal_discovery_model}")
    
def postprocess_graph(predicted_graph,
                      label_names,  
                      causal_discovery_model,
                      causal_discovery_type,
                      causal_discovery_library):
    
    # postprocess the graph
    if causal_discovery_model == 'ges':
        # Record[‘G’].graph[j,i] = 1 and Record[‘G’].graph[i,j] = -1 indicate i –> j; 
        # Record[‘G’].graph[i,j] = Record[‘G’].graph[j,i] = -1 indicates i — j.
        G = torch.tensor(predicted_graph['G'].graph)
    elif causal_discovery_model == 'grasp':
        G = torch.tensor(predicted_graph.graph)
    elif causal_discovery_model == "pc" and causal_discovery_library == 'causallearn':
        G = torch.tensor(predicted_graph.G.graph)
    elif causal_discovery_library == 'pytetrad':
        java_nodes = predicted_graph.getNodes()
        java_edges = predicted_graph.getEdges()
        nodes = [str(node.getName()) for node in java_nodes]
        adj_matrix = np.zeros((len(nodes), len(nodes)))
        for edge in java_edges:
            node_1 = edge.getNode1().getName().toString()
            node_2 = edge.getNode2().getName().toString()
            edge_end_point_1 = edge.getEndpoint1().toString()
            edge_end_point_2 = edge.getEndpoint2().toString()
            # edge_end_point_1 = 'tail' and edge_end_point_2 = 'head' indicates i -> j
            if edge_end_point_1 == 'TAIL' and edge_end_point_2 == 'ARROW':
                adj_matrix[nodes.index(node_1), nodes.index(node_2)] = 1
            # edge_end_point_1 = 'head' and edge_end_point_2 = 'tail' indicates i <- j
            elif edge_end_point_1 == 'ARROW' and edge_end_point_2 == 'TAIL':
                adj_matrix[nodes.index(node_2), nodes.index(node_1)] = 1
            # edge_end_point_1 = 'tail' and edge_end_point_2 = 'tail' indicates i - j
            elif edge_end_point_1 == 'TAIL' and edge_end_point_2 == 'TAIL':
                adj_matrix[nodes.index(node_1), nodes.index(node_2)] = -1
                adj_matrix[nodes.index(node_2), nodes.index(node_1)] = -1      
        G = torch.tensor(adj_matrix)
        label_names = nodes
    else:
        raise ValueError(f"Unknown causal discovery model: {causal_discovery_model}")
    
    if causal_discovery_library == 'causallearn':
        # redirect edges: [i,j]= 1 indicates i -> j; 
        #                 [i,j]= 1 and [j,i]= 1 indicates i - j;
        diff = G - G.T
        G[diff==-2] = 1
        G[diff== 2] = 0

    adj = pd.DataFrame(G.numpy(), index=label_names, columns=label_names, dtype=int)  
    return adj

def causal_discovery(cfg, dataset, true_graph=None):
    if cfg.causal_discovery is not None:
        if cfg.causal_discovery.name == 'llm':
            labels_names = dataset.c_info['names'] + dataset.y_info['names']
            dummy_matrix = np.zeros((len(labels_names), len(labels_names)))
            # set the elements which are not in the main diagonal to -1
            dummy_matrix[np.triu_indices(len(labels_names), k=1)] = -1
            dummy_matrix[np.tril_indices(len(labels_names), k=-1)] = -1
            predicted_graph = pd.DataFrame(dummy_matrix,
                                           index=labels_names, 
                                           columns=labels_names, 
                                           dtype=int)  
        else:
            print('Computing causal graph...')
            # Estimate causal graph with causal structural learning algorithms

            data_for_causal_discovery = process_data_for_causal_discovery(dataset.data['train'], 
                                                                        dataset.c_info['names'] +  dataset.y_info['names'],
                                                                        cfg.causal_discovery.get('causal_discovery_library'))

            raw_predicted_graph, model_info = apply_causal_discovery(data_for_causal_discovery,
                                                                    cfg.causal_discovery.get('name'),
                                                                    cfg.causal_discovery.get('type'),
                                                                    cfg.causal_discovery.get('causal_discovery_library'),
                                                                    **cfg.causal_discovery.get('kwargs', {}))
            
            predicted_graph = postprocess_graph(raw_predicted_graph,
                                                dataset.c_info['names'] + dataset.y_info['names'],
                                                cfg.causal_discovery.get('name'),
                                                cfg.causal_discovery.get('type'),
                                                cfg.causal_discovery.get('causal_discovery_library'))

            maybe_plot_graph(predicted_graph, 'predicted_graph')
            print('done')
        return predicted_graph
    else:
        return true_graph
    