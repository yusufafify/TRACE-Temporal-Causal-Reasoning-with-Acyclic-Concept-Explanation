import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from src.models.layers.base import MLP
from src.models.layers.c_encoder import ConceptBlock

class CEM(nn.Module):
    """
    Adaptation of the Concept Embedding Model (CEM) (https://arxiv.org/abs/2209.09056)
    """
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 concept_hidden_size,
                 output_size=2,
                 n_layers_encoder=1,
                 n_layers_concept_encoder=1,
                 n_layers_decoder=1,
                 activation='leaky_relu',
                 concept_loss_weight=0.5,
                 normalize_concept_loss=False,
                 dropout=0.0,
                 y_class_weights=None,
                 c_info={},
                 y_info={}):
        super(CEM, self).__init__()

        # to be stored for every model
        self.has_concepts = True
        self.is_causal = False

        # concepts info
        self.concept_names = c_info['names']
        self.virtual_roots = [name for name in c_info['names'] if name.startswith('#virtual_')]
        self.concept_loss_weight = concept_loss_weight
        if y_class_weights is not None:
            self.register_buffer('y_class_weights', torch.Tensor(y_class_weights))
        else:
            self.y_class_weights = None
        self.normalize_concept_loss = normalize_concept_loss

        # Encoder
        self.encoder = MLP(input_size=input_size,
                           hidden_size=hidden_size,
                           n_layers=n_layers_encoder,
                           activation=activation)
        
        # Concept encoders, one for each concept
        self.concept_encoders = nn.ModuleDict()
        for name in self.concept_names:
            if name in self.virtual_roots: continue
            self.concept_encoders[name] = ConceptBlock(input_size=hidden_size,
                                                       hidden_size=concept_hidden_size,
                                                       n_layers=n_layers_concept_encoder,
                                                       activation=activation,
                                                       c_cardinality=c_info['cardinality'][c_info['names'].index(name)])
        # Decoder
        if output_size is not None:
            n_concepts = len(self.concept_names) - len(self.virtual_roots)
            self.decoder = MLP(input_size=n_concepts*concept_hidden_size,
                               hidden_size=n_concepts*concept_hidden_size//2,
                               output_size=output_size,
                               n_layers=n_layers_decoder,
                               activation=activation,
                               dropout=dropout)
    
    
    def forward(self, x, c=None, intervention_index=None):
        """Forward pass of the model.
        Args:
            x (torch.Tensor): Input data
            c (torch.Tensor): Concept labels
            intervention_index (torch.Tensor): Intervention index
        Retrun:
            y_hat_probs (torch.Tensor): Predicted task probabilities
            c_hat_probs (Dict): Predicted concept probabilities
        """
        # Encode input, get the latent features
        x_encoded = self.encoder(x)

        # create embeddings and probabilities for each concept
        c_embeddings, c_hat_probs = {}, {}
        for name in self.concept_names:
            if name in self.virtual_roots: continue
            i = self.concept_names.index(name)
            c_embeddings[name], c_hat_probs[name] = self.concept_encoders[name](x_encoded, 
                                                                                c[:,i], 
                                                                                intervention_index[:,i])
        # Concatenate all concept embeddings
        concat_embeddings = torch.cat(list(c_embeddings.values()), dim=1)
        # Decode, get task logits
        y_hat_logits = self.decoder(concat_embeddings)
        y_hat_probs = torch.softmax(y_hat_logits, dim=1)
        return y_hat_probs, c_hat_probs
    

    def filter_output_for_loss(self, y_output, c_output):
        """Filter output for loss function"""
        return y_output, c_output
    
    def filter_output_for_metric(self, y_output, c_output):
        """Filter output for metric function"""
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

        # -- task loss: focal loss (γ=2.0) + label smoothing (ε=0.1)
        gamma = 2.0
        label_smoothing = 0.1
        n_classes = y_hat.size(1)
        y_hat_clamped = y_hat.clamp(min=1e-6, max=1.0 - 1e-6)
        log_probs = torch.log(y_hat_clamped)
        one_hot = torch.zeros_like(y_hat_clamped).scatter_(1, y.unsqueeze(1), 1.0)
        smooth_target = (1.0 - label_smoothing) * one_hot + label_smoothing / n_classes
        p_t = y_hat_clamped.gather(1, y.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** gamma
        nll = -(smooth_target * log_probs).sum(dim=1)
        if getattr(self, 'y_class_weights', None) is not None:
            alpha_t = self.y_class_weights[y]
            focal_nll = alpha_t * focal_weight * nll
        else:
            focal_nll = focal_weight * nll
        task_loss = focal_nll.mean()

        # -- concepts loss
        concept_loss = 0
        for name, c_hat in c_hat_dict.items():
            c_hat = torch.log(c_hat + 1e-6)
            concept_loss += loss_form(c_hat, c[:,self.concept_names.index(name)].long())
        if self.normalize_concept_loss:
            concept_loss /= len(c_hat_dict)
        
        return self.concept_loss_weight * concept_loss + (1-self.concept_loss_weight) * task_loss