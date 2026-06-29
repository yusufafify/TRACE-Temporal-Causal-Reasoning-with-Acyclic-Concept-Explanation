import torch
import torch.nn as nn
from src.models.layers.base import Dense, MLP
from src.models.layers.c_encoder import ConceptBlock
from src.models.layers.intervention import maybe_intervene
from src.utils import get_graph_levels, get_parents


class C2BM(nn.Module):
    """
    Causal version of the CEM model. 
    It propagates the information through a predefined causal graph.
    """
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 concept_hidden_size,
                 output_size=2,
                 n_layers_encoder=1,
                 n_layers_concept_encoder=1,
                 n_layers_propagation=1,
                 activation='leaky_relu',
                 concept_loss_weight=0.5,
                 normalize_concept_loss=False,
                 c_info={},
                 y_info={},
                 graph=None,
                 graph_labels=None,
                 prop_type='linear',
                 cat_latent=False,
                 y_class_weights=None,
                 concept_weights=None,
                 task_loss_type='cross_entropy',
                 input_encoder_type='mlp',
                 medicalnet_root=None,
                 medicalnet_pretrained_path=None,
                 medicalnet_in_channels=4,
                 medicalnet_seg_in_channels=3,
                 medicalnet_use_segmentation=True,
                 medicalnet_clinical_dim=0,
                 medicalnet_output_dim=None,
                 medicalnet_freeze_backbone=False,
                 medicalnet_sample_size=128,
                 scalar_passthrough_nodes=None,
                 deterministic_nodes=None,
                 concept_means=None,
                 concept_stds=None,
                 aux_binary_weight=0.0,
                 propagator_l1_weight=0.0,
                 cached_embedding_dim=128):
        super(C2BM, self).__init__()

        # to be stored for every model
        self.has_concepts = True
        self.is_causal = True
        self.normalize_concept_loss = normalize_concept_loss
        self.task_loss_type = task_loss_type
        # Deep-supervision: BCE on the PD probability of the 4-class output.
        # Adds a strong gradient signal toward the rare PD class without
        # changing the architecture. Ignored when output_size != 4.
        self.aux_binary_weight = float(aux_binary_weight)
        # L1 on the first linear layer of each propagator (sparse parent reliance).
        self.propagator_l1_weight = float(propagator_l1_weight)

        # define concepts info parameters
        self.c_names = c_info['names'] # used later to retrieve which are concepts
        self.y_names = y_info['names'] # and which are targerts
        self.virtual_roots = [name for name in c_info['names'] if name.startswith('#virtual_')]
        # Concepts that should be fed straight from the batch GT scalar to Y
        # (skip ConceptBlock + propagator). Use for metadata-style inputs like
        # time_gap that cannot be predicted from the image.
        self.scalar_passthrough_nodes = set(scalar_passthrough_nodes or [])
        # Concepts computed deterministically from parent predictions via closed-form
        # math (deltas, RANO threshold flags). No ConceptBlock, no learned propagator,
        # no concept loss. Closes embedding leakage on derived nodes.
        self.deterministic_nodes = set(deterministic_nodes or [])
        # z-score round-trip needs train-fold mean/std for each concept
        if concept_means is not None and concept_stds is not None:
            self.register_buffer('concept_means', torch.tensor(list(concept_means), dtype=torch.float32))
            self.register_buffer('concept_stds',  torch.tensor(list(concept_stds),  dtype=torch.float32))
        else:
            self.concept_means = None
            self.concept_stds = None
        self.combo_info = {'names': c_info['names'] + y_info['names'],
                           'cardinality': c_info['cardinality'] + y_info['cardinality']}
        assert self.c_names + self.y_names == graph_labels

        # Encoder
        self.input_encoder_type = input_encoder_type
        if input_encoder_type == 'cached':
            # Precomputed MedicalNet (or other) embeddings: identity passthrough.
            # Expected input shape: [B, cached_embedding_dim].
            self.encoder = torch.nn.Identity()
            encoder_output_size = int(cached_embedding_dim)
        elif input_encoder_type == 'medicalnet':
            from src.models.medicalnet_encoder import MedicalNetC2BMEncoder
            medicalnet_kwargs = dict(
                output_dim=medicalnet_output_dim if medicalnet_output_dim is not None else hidden_size,
                in_channels=medicalnet_in_channels,
                seg_in_channels=medicalnet_seg_in_channels,
                use_segmentation=medicalnet_use_segmentation,
                clinical_dim=medicalnet_clinical_dim,
                freeze_backbone=medicalnet_freeze_backbone,
                sample_size=medicalnet_sample_size,
            )
            if medicalnet_root is not None:
                medicalnet_kwargs['medicalnet_root'] = medicalnet_root
            if medicalnet_pretrained_path is not None:
                medicalnet_kwargs['pretrained_path'] = medicalnet_pretrained_path
            self.encoder = MedicalNetC2BMEncoder(**medicalnet_kwargs)
            encoder_output_size = self.encoder.output_dim
        else:
            self.encoder = MLP(input_size=input_size,
                               hidden_size=hidden_size,
                               n_layers=n_layers_encoder,
                               activation=activation)
            encoder_output_size = hidden_size

        # Concept encoders, one for each concept (skip pass-through and
        # deterministic nodes — they have no learnable parameters).
        self.concept_encoders = nn.ModuleDict()
        for name in self.combo_info['names']:
            if name in self.scalar_passthrough_nodes or name in self.deterministic_nodes:
                continue
            self.concept_encoders[name] = ConceptBlock(input_size=encoder_output_size,
                                                       hidden_size=concept_hidden_size,
                                                       n_layers=n_layers_concept_encoder,
                                                       activation=activation,
                                                       c_cardinality=self.combo_info['cardinality'][self.combo_info['names'].index(name)])
        self.concept_hidden_size = concept_hidden_size
        self.concept_loss_weight = concept_loss_weight
        
        if y_class_weights is not None:
            self.register_buffer('y_class_weights', torch.Tensor(y_class_weights))
        else:
            self.y_class_weights = None

        if concept_weights is not None:
            self.register_buffer('concept_weights', torch.Tensor(concept_weights))
        else:
            self.concept_weights = None

        # get levels
        self.graph = torch.Tensor(graph).int()
        task_index = self.combo_info['names'].index(self.y_names[0])
        graph_levels = get_graph_levels(self.graph, task_index)
        self.roots = graph_levels[0]
        self.roots_info = {'names': [name for i, name in enumerate(self.combo_info['names']) 
                                     if i in self.roots], 
                           'cardinality': [card for i, card in enumerate(self.combo_info['cardinality']) 
                                           if i in self.roots]}
        if self.y_names[0] in self.roots_info['names']:
            raise ValueError('The target variable cannot be a root concept')
        
        # get list of propagators
        self.prop_type = prop_type
        self.cat_latent = cat_latent
        # For deterministic nodes, remember the level so forward processes them
        # in topological order (after their parents are populated).
        self._deterministic_levels = {}  # {level_str: [node_name, ...]}

        def _parent_feat_dim(parent_name):
            """c_features dim contributed by a parent at propagator-input time.
            Passthrough and deterministic nodes both emit cardinality-sized
            features (1 for continuous, 2 for binary flags) — no embedding pipe.
            All other (learned) nodes contribute concept_hidden_size."""
            if parent_name in self.scalar_passthrough_nodes or parent_name in self.deterministic_nodes:
                pidx = self.combo_info['names'].index(parent_name)
                return self.combo_info['cardinality'][pidx]
            return concept_hidden_size

        if prop_type in ['dense', 'mlp', 'embeddings', 'equations']:
            self.propagators = nn.ModuleDict()
            for i in range(1, len(graph_levels)):
                level = graph_levels[i]
                self.propagators[str(i)] = nn.ModuleDict()
                for node in level:
                    node_name = self.combo_info['names'][node]
                    if node_name in self.deterministic_nodes:
                        self._deterministic_levels.setdefault(str(i), []).append(node_name)
                        continue  # no learned propagator for deterministic nodes
                    parents = get_parents(self.graph, node).tolist()
                    node_cardinality = self.combo_info['cardinality'][node]
                    parents_cardinality = [self.combo_info['cardinality'][p] for p in parents]
                    parent_names = [self.combo_info['names'][p] for p in parents]
                    parents_feat_total = sum(_parent_feat_dim(pn) for pn in parent_names)
                    if prop_type == 'dense':
                        prop_input_size = parents_feat_total + (encoder_output_size if cat_latent else 0)
                        self.propagators[str(i)][node_name] = Dense(input_size = prop_input_size,
                                                                    output_size = node_cardinality,
                                                                    activation = activation)
                    elif prop_type == 'mlp':
                        prop_input_size = parents_feat_total + (encoder_output_size if cat_latent else 0)
                        self.propagators[str(i)][node_name] = MLP(input_size = prop_input_size,
                                                                  hidden_size = concept_hidden_size,
                                                                  output_size = node_cardinality,
                                                                  n_layers = n_layers_propagation,
                                                                  activation = activation)
                    elif prop_type == 'embeddings':
                        self.propagators[str(i)][node_name] = MLP(input_size = parents_feat_total,
                                                                  hidden_size = concept_hidden_size,
                                                                  output_size = node_cardinality,
                                                                  n_layers = n_layers_propagation,
                                                                  activation = activation)
                        # TODO: embeddings for the latent factors if there are undirected edges
                    elif prop_type == 'equations':
                        self.propagators[str(i)][node_name] = MLP(input_size = node_cardinality*concept_hidden_size,
                                                                  hidden_size = concept_hidden_size,
                                                                  output_size = sum(parents_cardinality)*node_cardinality,
                                                                  n_layers = n_layers_propagation,
                                                                  activation = activation)
        else:
            raise ValueError('invalid prop_type')


    # ----------------------------------------------------------------- #
    # Deterministic propagators: closed-form math from parent predictions
    # ----------------------------------------------------------------- #
    def _to_raw_cm3(self, z_value, name):
        """Un-z-score and undo log1p to get a raw clinical value (cm^3 or cm^2)."""
        idx = self.c_names.index(name)
        log_v = z_value * self.concept_stds[idx] + self.concept_means[idx]
        return torch.expm1(log_v).clamp(min=0)

    def _z_score(self, raw_value, name):
        """Re-z-score a raw value back into bottleneck space."""
        idx = self.c_names.index(name)
        return (raw_value - self.concept_means[idx]) / (self.concept_stds[idx] + 1e-8)

    def _compute_deterministic(self, c_name, c, intervention_index, c_probs, c_features):
        """Populate c_probs[c_name] and c_features[c_name] for a deterministic node
        using closed-form math from parent predictions (which must already exist
        in c_probs). Supports maybe_intervene so the existing per-concept MoRF
        table keeps producing meaningful Δ F1 numbers on these nodes.
        """
        if self.concept_means is None or self.concept_stds is None:
            raise RuntimeError(
                f"Deterministic node '{c_name}' needs concept_means/concept_stds "
                "passed at construction time (z-score round-trip)."
            )
        EPS = 1e-4
        c_idx = self.c_names.index(c_name)

        if c_name == 'delta_enhancing_absolute':
            v_b = self._to_raw_cm3(c_probs['enhancing_tumor_volume_cm3'].squeeze(-1), 'enhancing_tumor_volume_cm3')
            v_f = self._to_raw_cm3(c_probs['followup_enhancing_volume_cm3'].squeeze(-1), 'followup_enhancing_volume_cm3')
            z = self._z_score(v_f - v_b, c_name).unsqueeze(-1)
            c_probs[c_name] = z
        elif c_name == 'delta_enhancing_percent':
            v_b = self._to_raw_cm3(c_probs['enhancing_tumor_volume_cm3'].squeeze(-1), 'enhancing_tumor_volume_cm3')
            v_f = self._to_raw_cm3(c_probs['followup_enhancing_volume_cm3'].squeeze(-1), 'followup_enhancing_volume_cm3')
            raw = ((v_f - v_b) / (v_b + EPS)).clamp(-1.0, 5.0)
            c_probs[c_name] = self._z_score(raw, c_name).unsqueeze(-1)
        elif c_name == 'delta_non_enhancing_absolute':
            v_b = self._to_raw_cm3(c_probs['non_enhancing_volume_cm3'].squeeze(-1), 'non_enhancing_volume_cm3')
            v_f = self._to_raw_cm3(c_probs['followup_non_enhancing_volume_cm3'].squeeze(-1), 'followup_non_enhancing_volume_cm3')
            c_probs[c_name] = self._z_score(v_f - v_b, c_name).unsqueeze(-1)
        elif c_name == 'delta_non_enhancing_percent':
            v_b = self._to_raw_cm3(c_probs['non_enhancing_volume_cm3'].squeeze(-1), 'non_enhancing_volume_cm3')
            v_f = self._to_raw_cm3(c_probs['followup_non_enhancing_volume_cm3'].squeeze(-1), 'followup_non_enhancing_volume_cm3')
            raw = ((v_f - v_b) / (v_b + EPS)).clamp(-1.0, 5.0)
            c_probs[c_name] = self._z_score(raw, c_name).unsqueeze(-1)
        elif c_name == 'delta_spd_absolute':
            v_b = self._to_raw_cm3(c_probs['baseline_spd_cm2'].squeeze(-1), 'baseline_spd_cm2')
            v_f = self._to_raw_cm3(c_probs['followup_spd_cm2'].squeeze(-1), 'followup_spd_cm2')
            c_probs[c_name] = self._z_score(v_f - v_b, c_name).unsqueeze(-1)
        elif c_name == 'delta_spd_percent':
            v_b = self._to_raw_cm3(c_probs['baseline_spd_cm2'].squeeze(-1), 'baseline_spd_cm2')
            v_f = self._to_raw_cm3(c_probs['followup_spd_cm2'].squeeze(-1), 'followup_spd_cm2')
            raw = ((v_f - v_b) / (v_b + EPS)).clamp(-1.0, 5.0)
            c_probs[c_name] = self._z_score(raw, c_name).unsqueeze(-1)
        elif c_name == 'vol_pd_flag':
            v_b = self._to_raw_cm3(c_probs['enhancing_tumor_volume_cm3'].squeeze(-1), 'enhancing_tumor_volume_cm3')
            v_f = self._to_raw_cm3(c_probs['followup_enhancing_volume_cm3'].squeeze(-1), 'followup_enhancing_volume_cm3')
            d_pct = ((v_f - v_b) / (v_b + EPS)).clamp(-1.0, 5.0)
            flag = ((d_pct >= 0.40) & (v_b >= 0.5)).float()
            c_probs[c_name] = torch.stack([1.0 - flag, flag], dim=1)
        elif c_name == 'vol_pr_flag':
            v_b = self._to_raw_cm3(c_probs['enhancing_tumor_volume_cm3'].squeeze(-1), 'enhancing_tumor_volume_cm3')
            v_f = self._to_raw_cm3(c_probs['followup_enhancing_volume_cm3'].squeeze(-1), 'followup_enhancing_volume_cm3')
            d_pct = ((v_f - v_b) / (v_b + EPS)).clamp(-1.0, 5.0)
            flag = ((d_pct <= -0.65) & (v_b >= 0.5)).float()
            c_probs[c_name] = torch.stack([1.0 - flag, flag], dim=1)
        elif c_name == 'spd_pd_flag':
            v_b = self._to_raw_cm3(c_probs['baseline_spd_cm2'].squeeze(-1), 'baseline_spd_cm2')
            v_f = self._to_raw_cm3(c_probs['followup_spd_cm2'].squeeze(-1), 'followup_spd_cm2')
            d_pct = ((v_f - v_b) / (v_b + EPS)).clamp(-1.0, 5.0)
            flag = ((d_pct >= 0.25) & (v_b >= 0.01)).float()
            c_probs[c_name] = torch.stack([1.0 - flag, flag], dim=1)
        elif c_name == 'spd_pr_flag':
            v_b = self._to_raw_cm3(c_probs['baseline_spd_cm2'].squeeze(-1), 'baseline_spd_cm2')
            v_f = self._to_raw_cm3(c_probs['followup_spd_cm2'].squeeze(-1), 'followup_spd_cm2')
            d_pct = ((v_f - v_b) / (v_b + EPS)).clamp(-1.0, 5.0)
            flag = ((d_pct <= -0.50) & (v_b >= 0.01)).float()
            c_probs[c_name] = torch.stack([1.0 - flag, flag], dim=1)
        else:
            raise NotImplementedError(
                f"No deterministic formula registered for '{c_name}'."
            )

        # Apply test-time intervention so the existing MoRF/LeRF table keeps
        # producing non-trivial Δ F1 on deterministic nodes.
        if c_name not in self.y_names:
            c_probs[c_name] = maybe_intervene(
                c_probs[c_name], c[:, c_idx], intervention_index[:, c_idx]
            )
        # c_features feeds downstream propagators; keep it as scalar/one-hot
        # so no x_encoded info is smuggled to Y through this node.
        c_features[c_name] = c_probs[c_name]

    def forward(self, x, c=None, intervention_index=None, x_baseline=None,
                x_seg_curr=None, x_seg_base=None, clinical_features=None,
                cached_x_encoded=None, return_x_encoded=False, **kwargs):
        # Encode input, get the latent features (or reuse a cached encoding so
        # downstream callers can run a second propagator pass without paying
        # the 3D encoder cost again — used by the intervention-consistency
        # second forward in Predictor.shared_step).
        if cached_x_encoded is not None:
            x_encoded = cached_x_encoded
        elif self.input_encoder_type == 'cached':
            # x already IS the embedding; identity encoder.
            x_encoded = x
        elif self.input_encoder_type == 'medicalnet':
            x_encoded = self.encoder(
                x,
                x_baseline,
                x_seg_curr=x_seg_curr,
                x_seg_base=x_seg_base,
                clinical_features=clinical_features,
            )
        else:
            x_encoded = self.encoder(x)

        c_embs, c_probs, c_values_emb, c_features = {}, {}, {}, {}
        for i, name in enumerate(self.combo_info['names']):
            # Scalar pass-through: feed batch GT directly, no ConceptBlock,
            # no embedding pipe. Closes the leakage path for metadata inputs
            # the image cannot predict (e.g. time_gap), and for label-derived
            # but mask-deterministic flags (e.g. new_lesion_flag, which is
            # already computed from segmentation masks during preprocessing).
            if name in self.scalar_passthrough_nodes:
                cardinality = self.combo_info['cardinality'][i]
                if cardinality == 1:
                    val = c[:, i].unsqueeze(-1)  # [B, 1] z-scored truth
                else:
                    # Binary/categorical: one-hot over GT class index
                    val_int = c[:, i].round().long().clamp(0, cardinality - 1)
                    val = torch.nn.functional.one_hot(val_int, num_classes=cardinality).float()
                c_probs[name] = val
                c_features[name] = val
                continue
            # Deterministic nodes: skip ConceptBlock; populated later from parents.
            if name in self.deterministic_nodes:
                continue
            # create embeddings and probabilities for each root concept
            # this assumes the task is last in the name list
            if name in self.roots_info['names']:
                c_embs[name], c_probs[name] =  self.concept_encoders[name](x_encoded,
                                                                           c[:,i],
                                                                           intervention_index[:,i],
                                                                           to_return=['embs', 'probs'])
                # For mlp/dense propagators: use concept_hidden_size-dim concept embeddings as features
                # (much richer than the 2-dim softmax probs)
                c_features[name] = c_embs[name]
            else:
                # create latent for each non-root concept    
                # remember not to intervene on the task
                c_values_emb[name] =  self.concept_encoders[name](x_encoded, 
                                                                  c[:,i] if name in self.c_names else None, 
                                                                  intervention_index[:,i] if name in self.c_names else None,
                                                                  to_return=['values_embs'])
                
        # propagate the information through the causal graph
        # loop over le graph levels, starting from the level 1 after the roots
        for level_id_str, level in self.propagators.items():
            # ---- learned propagators at this level ----
            # update all nodes in the level
            for c_name, propagator in level.items():
                c_index = self.combo_info['names'].index(c_name)
                p_indices = get_parents(self.graph, c_index).tolist()
                p_names = [self.combo_info['names'][p] for p in p_indices]
                
                if self.prop_type =='dense' or self.prop_type == 'mlp':
                    # propagate concept embeddings (richer than raw probs)
                    c_feat_parents = torch.cat([c_features[p_name] for p_name in p_names], dim=1)
                    if self.cat_latent: 
                        c_feat_parents = torch.cat([c_feat_parents, x_encoded], dim=1)
                    prop_out = propagator(c_feat_parents)
                    c_card = self.combo_info['cardinality'][c_index]
                    if c_card > 1:
                        c_probs[c_name] = torch.softmax(prop_out, dim=1)
                        # Re-compute embedding for this node from its concept encoder
                        # so downstream propagators get concept_hidden_size-dim features too
                        c_features[c_name] = c_embs.get(c_name) if c_name in c_embs else c_probs[c_name]
                    else:
                        c_probs[c_name] = prop_out
                        # Use concept encoder embedding × scalar for concept_hidden_size-dim features
                        if c_name in c_values_emb:
                            c_features[c_name] = c_values_emb[c_name] * prop_out  # [B, hidden] * [B, 1] → [B, hidden]
                        else:
                            c_features[c_name] = prop_out
                    if c_name not in self.y_names:
                        c_probs[c_name] = maybe_intervene(c_probs[c_name], c[:,c_index], intervention_index[:,c_index])
                        if c_card > 1:
                            # After intervention, recompute embedding if intervened
                            # For non-leaf non-root nodes, use the concept encoder to get embedding
                            if c_name in c_values_emb:
                                first = c_values_emb[c_name].reshape(-1, c_card, self.concept_hidden_size)
                                second = c_probs[c_name].unsqueeze(-1)
                                c_features[c_name] = (first * second).sum(dim=1)
                            else:
                                c_features[c_name] = c_probs[c_name]
                        else:
                            # Cardinality 1: embedding × scalar for downstream propagation
                            if c_name in c_values_emb:
                                c_features[c_name] = c_values_emb[c_name] * c_probs[c_name]
                            else:
                                c_features[c_name] = c_probs[c_name]

                elif self.prop_type == 'embeddings':
                    c_cardinality = self.combo_info['cardinality'][c_index]
                    # propagate embeddings
                    c_embs_parents = torch.cat([c_embs[p_name] for p_name in p_names], dim=1)
                    c_probs[c_name] = torch.softmax(propagator(c_embs_parents), dim=1)
                    if c_name not in self.y_names:
                        c_probs[c_name] = maybe_intervene(c_probs[c_name], c[:,c_index], intervention_index[:,c_index])
                    # compute the weighted average between concepts embedding and probabilities
                    first = c_values_emb[c_name].reshape(-1, c_cardinality, self.concept_hidden_size)
                    second = c_probs[c_name].unsqueeze(-1)
                    c_embs[c_name] = (first * second).sum(dim=1)

                elif self.prop_type == 'equations':
                    c_cardinality = self.combo_info['cardinality'][c_index]
                    p_cardinality = [self.combo_info['cardinality'][p] for p in p_indices]
                    # propagate embeddings
                    c_prop_parents = torch.cat([c_probs[p_name] for p_name in p_names], dim=1).unsqueeze(-1)
                    weights = propagator(c_values_emb[c_name])
                    weights = weights.reshape(-1, c_cardinality, sum(p_cardinality))
                    c_probs[c_name] = torch.softmax(torch.matmul(weights, c_prop_parents).squeeze(-1), dim=1)
                    if c_name not in self.y_names:
                        c_probs[c_name] = maybe_intervene(c_probs[c_name], c[:,c_index], intervention_index[:,c_index])
                    # first = c_values_emb[c_name].reshape(-1, c_cardinality, self.concept_hidden_size)
                    # second = c_probs[c_name].unsqueeze(-1)
                    # c_embs[c_name] = (first * second).sum(dim=1)

            # ---- deterministic nodes at this level (closed-form math) ----
            for c_name in self._deterministic_levels.get(level_id_str, []):
                self._compute_deterministic(c_name, c, intervention_index, c_probs, c_features)

        # Decode, get task logits
        y_hat_probs = c_probs[self.y_names[0]]
        # filter virtual roots
        c_hat_probs = {k:v for k,v in c_probs.items() if k in self.c_names and k not in self.virtual_roots}
        if return_x_encoded:
            return y_hat_probs, c_hat_probs, x_encoded
        return y_hat_probs, c_hat_probs
    
    def filter_output_for_loss(self, y_output, c_output):
        """Filter output for loss function"""
        return y_output, c_output
    
    def filter_output_for_metric(self, y_output, c_output):
        """Filter output for metric function"""
        return y_output, c_output

    @staticmethod
    def _macro_soft_f1(probs, targets, n_classes, epsilon=1e-7):
        """Differentiable approximation of 1 - macro F1."""
        targets_oh = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
        tp = (probs * targets_oh).sum(dim=0)
        fp = (probs * (1 - targets_oh)).sum(dim=0)
        fn = ((1 - probs) * targets_oh).sum(dim=0)
        precision = tp / (tp + fp + epsilon)
        recall    = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        return 1.0 - f1.mean()

    def loss(self, y_hat, y, c_hat_dict, c):
        """Compute loss function.
        Args:
            y_hat (torch.Tensor): Predicted task probabilities
            y (torch.Tensor): True task labels
            c_hat_dict (Dict): Predicted concept probabilities
            c (torch.Tensor): True concept labels"""
        y = y.flatten().long()

        n_classes = y_hat.size(1)
        y_hat_clamped = y_hat.clamp(min=1e-6, max=1.0 - 1e-6)
        log_probs = torch.log(y_hat_clamped)

        if self.task_loss_type in ['cross_entropy', 'ce']:
            # y_hat is already a probability distribution, so NLL on log-probs
            # is the CrossEntropyLoss equivalent for this model output.
            task_loss = torch.nn.functional.nll_loss(
                log_probs, y, weight=self.y_class_weights
            )
        else:
            # Focal + soft macro-F1 legacy task loss.
            gamma = 2.0
            label_smoothing = 0.1

            one_hot = torch.zeros_like(y_hat_clamped).scatter_(1, y.unsqueeze(1), 1.0)
            smooth_target = (1.0 - label_smoothing) * one_hot + label_smoothing / n_classes
            p_t = y_hat_clamped.gather(1, y.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - p_t) ** gamma
            nll = -(smooth_target * log_probs).sum(dim=1)

            if self.y_class_weights is not None:
                alpha_t = self.y_class_weights[y]
                focal_nll = alpha_t * focal_weight * nll
            else:
                focal_nll = focal_weight * nll

            focal_loss = focal_nll.mean()
            soft_f1_loss = self._macro_soft_f1(y_hat_clamped, y, n_classes)
            task_loss = 0.5 * focal_loss + 0.5 * soft_f1_loss

        # -- aux binary deep supervision on PD probability (4-class only)
        if self.aux_binary_weight > 0 and n_classes == 4:
            pd_prob = y_hat_clamped[:, 3].float()
            pd_target = (y == 3).float()
            # Stay outside autocast: BCE on probs is autocast-unsafe in bf16/fp16.
            with torch.cuda.amp.autocast(enabled=False):
                aux_bce = torch.nn.functional.binary_cross_entropy(pd_prob, pd_target)
            task_loss = task_loss + self.aux_binary_weight * aux_bce

        # -- concepts loss
        concept_loss = 0
        for ci, (name, c_hat) in enumerate(c_hat_dict.items()):
            # Pass-through nodes have no learnable params for the concept;
            # their "prediction" equals GT by construction, so no loss term.
            if name in self.scalar_passthrough_nodes:
                continue
            # Deterministic nodes have no learnable params either; their value
            # is fully determined by their parents' predicted volumes.
            if name in self.deterministic_nodes:
                continue
            c_index = self.c_names.index(name)
            c_raw = c[:, c_index]
            if c_hat.shape[1] == 1:
                # Regression concept: MSE on z-scored values
                # Clip both target and prediction to a reasonable sigma range
                # to prevent a few large outliers from dominating the loss
                c_target_clipped = c_raw.clamp(-5.0, 5.0)
                c_hat_clipped = c_hat.squeeze(-1).clamp(-5.0, 5.0)
                mse_loss = torch.nn.functional.mse_loss(c_hat_clipped, c_target_clipped)
                concept_loss += mse_loss
            else:
                # Classification concept: use NLL loss with class targets
                c_target = c_raw.long()
                
                # Determine how many classes this concept has
                n_cls = c_hat.shape[1]
                
                # Use fixed weights if available, else compute on the fly (fallback)
                if self.concept_weights is not None and c_index < self.concept_weights.shape[0]:
                    weights = self.concept_weights[c_index][:n_cls]
                else:
                    # Inverse frequency fallback
                    weights = torch.ones(n_cls, device=c_hat.device)
                    for cls_idx in range(n_cls):
                        n_c = (c_target == cls_idx).float().sum().clamp(min=1)
                        weights[cls_idx] = float(len(c_target)) / (n_cls * n_c)
                
                c_hat_log = torch.log(c_hat + 1e-6)
                concept_loss += torch.nn.functional.nll_loss(c_hat_log, c_target, weight=weights)
            
        if self.normalize_concept_loss:
            concept_loss /= len(c_hat_dict)

        total = self.concept_loss_weight * concept_loss + task_loss

        if self.propagator_l1_weight > 0 and hasattr(self, 'propagators'):
            l1 = 0.0
            n = 0
            for level in self.propagators.values():
                for prop in level.values():
                    first_linear = None
                    for m in prop.modules():
                        if isinstance(m, torch.nn.Linear):
                            first_linear = m
                            break
                    if first_linear is not None:
                        l1 = l1 + first_linear.weight.abs().mean()
                        n += 1
            if n > 0:
                total = total + self.propagator_l1_weight * (l1 / n)

        return total
