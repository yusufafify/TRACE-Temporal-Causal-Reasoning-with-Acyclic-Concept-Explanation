from env import CACHE
import os
import numpy as np
import torch
from torch_geometric.utils import to_dense_adj
import bnlearn as bn
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling

from src.data.utils import split_dataset

class BNDataset():
    def __init__(self, 
                 dag_name = 'asia',
                 task_name = 'dysp',
                 dataset_n_samples: int = 10000,
                 test_size:  float = 0.2, # proportion of the dataset to include in the test set
                 val_size: float = 0.1, # proportion of the training set to include in the validation set
                 ftune_size: float = 0., # proportion of the test set to include in the finetuning set
                 ftune_val_size: float = 0., # proportion of the finetuning set to include in the finetuning validation set
                 bias: dict = {'train': {'mode': False, 'kwargs': {}},
                               'test': {'mode': False, 'kwargs': {}}},
                 to_keep: list = None):
        
        self.to_keep = to_keep
        self.dag_name = dag_name
        if dag_name in ['asia', 'alarm', 'andes', 'sachs', 'water']:
            bn_model_dict = bn.import_DAG(self.dag_name)
        else:
            path = os.path.join(CACHE, 'bnlearn_bif_datasets', f'{dag_name}.bif')
            bn_model_dict = bn.import_DAG(path)
        self.bn_model_dict = bn_model_dict
        self.bn_model = bn_model_dict["model"]
        
        self.dataset_n_samples = dataset_n_samples
        self.test_size = test_size
        self.val_size = val_size
        self.ftune_size = ftune_size

        self.bias = bias
        
        if self.to_keep is not None:
            c_names = [name for name in list(self.bn_model.nodes()) 
                       if name != task_name and name in self.to_keep]  # All except the last node
        else:
            c_names = [name for name in list(self.bn_model.nodes()) if name != task_name]  # All except the last node
        y_name = [task_name]
        c_cardinalities = [int(self.bn_model.get_cardinality()[node]) for node in c_names]  # Cardinalities for all c_names
        y_cardinality = [int(self.bn_model.get_cardinality()[y_name[0]])]

        # Complete bottleneck
        self.complete_bottleneck_names = [name for name in list(self.bn_model.nodes()) if name != task_name]
        self.complete_c_cardinalities = [int(self.bn_model.get_cardinality()[node]) \
                                         for node in self.complete_bottleneck_names]  # Cardinalities for all c_names

        self.c_info = {'names': c_names, 
                       'cardinality': c_cardinalities}
        self.c_info_complete = {'names': self.complete_bottleneck_names,
                                'cardinality': self.complete_c_cardinalities}
        self.y_info = {'names': y_name,
                       'cardinality': y_cardinality}  
        self.data = {}
        
        
    def load_ground_truth_graph(self): 
        node_labels = self.c_info['names'] + self.y_info['names']
        # reorder the adjacency matrix to match the new node labels
        adj_pandas = self.bn_model_dict['adjmat'].loc[node_labels, node_labels]
        if self.to_keep is not None:
            # remove the nodes that are not in the to_keep list
            adj_pandas = adj_pandas.loc[self.to_keep, self.to_keep]
            adj_pandas = adj_pandas.loc[:, self.to_keep]
        # convert the pandas dataframe to a numpy array
        self.adj = adj_pandas.astype(int)
        return self.adj
    
    def split(self):
        """ 
        Split the dataset into training, validation and test sets 
        """
        # self.bn_model is biased after this point
        self.data['train'] = _BNDataset(bn_model = self.bn_model,
                                        n_samples = int(self.dataset_n_samples*(1-self.val_size-self.test_size)),
                                        task_name = self.y_info['names'][0],
                                        bias_mode = self.bias['train'].get('mode'),
                                        bias_kwargs = self.bias['train'].get('kwargs'),
                                        to_keep=self.to_keep)
        self.data['val'] = _BNDataset(bn_model = self.bn_model,
                                      n_samples = int(self.dataset_n_samples*self.val_size),
                                      task_name = self.y_info['names'][0],
                                      bias_mode = self.bias['train'].get('mode'),
                                      bias_kwargs = self.bias['train'].get('kwargs'),
                                      to_keep=self.to_keep)
        self.data['test'] = _BNDataset(bn_model = self.bn_model,
                                       n_samples = int(self.dataset_n_samples*self.test_size),
                                       task_name = self.y_info['names'][0],
                                       bias_mode = self.bias['test'].get('mode'),
                                       bias_kwargs = self.bias['test'].get('kwargs'),
                                       to_keep=self.to_keep)
        self.data['train'].split_type = 'train'
        self.data['val'].split_type = 'val'
        if self.ftune_size > 0:
            self.data['test'], self.data['ftune'] = split_dataset(self.data['test'], self.ftune_size)
            self.data['ftune'].split_type = 'ftune'


class _BNDataset(torch.utils.data.Dataset):

    def __init__(self,
                    bn_model: dict,
                    n_samples: int, 
                    task_name: str,
                    bias_mode: str = 'random',
                    bias_kwargs: dict = {},
                    to_keep: list = None):
        
        super().__init__()

        self.to_keep = to_keep
        self.bn_model = bn_model.copy()
        self.n_samples = n_samples
        self.bias_mode = bias_mode
        self.bias_kwargs = bias_kwargs
        self.split_type = ""
        self.graph = []

        self._generate_biased_data()
        
        if self.to_keep is not None:
            concept_names = [name for name in list(self.data.columns) 
                             if name != task_name and name in self.to_keep]
        else:
            concept_names = [name for name in list(self.data.columns) if name != task_name]
        
        self.complete_bottleneck_names = [name for name in list(self.data.columns) if name != task_name]
        
        reordered_names = concept_names + [task_name]
        self.y = torch.Tensor(self.data.loc[:,task_name].values).float().unsqueeze(1)
        self.c = torch.Tensor(self.data.loc[:,concept_names].values).float()
        self.complete_c = torch.Tensor(self.data.loc[:,self.complete_bottleneck_names].values).float()
        self.X = torch.Tensor(self.data.loc[:,reordered_names].values).float()

    def _generate_biased_data(self):
        if self.bias_mode == False:
            inference = BayesianModelSampling(self.bn_model)
            self.data = inference.forward_sample(size=self.n_samples)
        elif self.bias_mode == 'custom':
            for arc, probs in zip(self.bias_kwargs['add_arc'], self.bias_kwargs['add_arc_prob']):
                self.bn_model.add_edge(arc[0], arc[1])
                evidence_card = [self.bn_model.get_cardinality()[node] for node in self.bn_model.get_parents(arc[1])]
                new_cpd = TabularCPD(variable=arc[1], 
                                     variable_card=self.bn_model.get_cardinality()[arc[1]],  # binary variable (0 or 1)
                                     values=probs, 
                                     evidence=self.bn_model.get_parents(arc[1]), 
                                     evidence_card=evidence_card) 
                self.bn_model.add_cpds(new_cpd)
            inference = BayesianModelSampling(self.bn_model)
            self.data = inference.forward_sample(size=self.n_samples)
        else:
            raise ValueError(f"Unknown bias mode: {self.bias_mode}")    

    def register_graph(self, graph):
        self.graph = graph

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        c = self.c[index]
        complete_c = self.complete_c[index]
        y = self.y[index]
        return {'x':x, 'c':c, 'complete_c':complete_c, 'y':y, 'graph':self.graph}
    