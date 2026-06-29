import torch
import torch.nn as nn

from src.models.layers.base import MLP

class BlackBox_Multi(nn.Module):
    """
    Opaque model with multiple outputs. It predicts both task and concept labels.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=2,
                 n_layers=1,
                 activation='leaky_relu',
                 dropout=0.0,
                 c_info={},
                 y_info={}):
        super(BlackBox_Multi, self).__init__()
        
        # to be stored for every model
        self.has_concepts = False
        self.is_causal = False
        
        self.concept_names = c_info['names']
        self.concept_cardinality = c_info['cardinality']
        self.task_cardinality = y_info['cardinality']
        self.virtual_roots = [name for name in c_info['names'] if name.startswith('#virtual_')]
        vr_cardinality = [c_info['cardinality'][c_info['names'].index(name)] for name in self.virtual_roots]

        output_size = output_size + sum(self.concept_cardinality) - sum(vr_cardinality)

        self.mlp = MLP(input_size=input_size,
                       hidden_size=hidden_size,
                       output_size=output_size,
                       n_layers=n_layers,
                       activation=activation,
                       dropout=dropout)


    def forward(self, x, c=None, intervention_index=None):
        all_hat_logits = self.mlp(x)
        y_hat_probs = torch.softmax(all_hat_logits[:, :self.task_cardinality[0]], dim=1)
        c_hat_probs = {}
        for i, name in enumerate(self.concept_names):
            if name in self.virtual_roots: continue
            c_hat_probs[name] = torch.softmax(all_hat_logits[:, self.task_cardinality[0] + sum(self.concept_cardinality[:i]):
                                                               self.task_cardinality[0] + sum(self.concept_cardinality[:i+1])], dim=1)
        return y_hat_probs, c_hat_probs
    
    def filter_output_for_loss(self, y_output, c_output):
        return y_output, c_output
    
    def filter_output_for_metric(self, y_output, c_output):
        return y_output, c_output
    
    def loss(self, y_hat, y, c_hat_dict, c):
        """Compute loss function.
        Args:
            y_hat (torch.Tensor): Predicted task probabilities
            y (torch.Tensor): True task labels
            c_hat_dict (Dict): Predicted concept probabilities
            c (torch.Tensor): True concept labels"""
        y = y.flatten().long()
        # c = c.long() # later to avoid nan to disappear
        loss_form = torch.nn.NLLLoss()

        # -- task loss
        y_hat = torch.log(y_hat + 1e-6)
        task_loss = loss_form(y_hat, y)

        # -- concepts loss
        concept_loss = 0
        for name, c_hat in c_hat_dict.items():
            c_hat = torch.log(c_hat + 1e-6)
            concept_loss += loss_form(c_hat, c[:,self.concept_names.index(name)].long())

        return concept_loss + task_loss