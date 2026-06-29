from typing import Any, Optional, Mapping, Type
import pickle
import itertools
import numpy as np

import torch
from torch import nn
from torchmetrics import Metric, MetricCollection
from torchmetrics.collections import _remove_prefix
import pytorch_lightning as pl

from src.models.layers.intervention import get_test_intervention_index

class Predictor(pl.LightningModule):    
    def __init__(self,
                model: Optional[nn.Module] = None,
                metrics: Optional[Mapping[str, Metric]] = None,
                optim_class: Optional[Type] = None,
                optim_kwargs: Optional[Mapping] = None,
                scheduler_class: Optional[Type] = None,
                scheduler_kwargs: Optional[Mapping] = None,
                intervention_prob: Optional[float] = 0.2,
                c_names: Optional[list] = None,
                test_interv_policy: Optional[str] = None,
                test_interv_noise: Optional[float] = 0.,
                propagator_warmup_epochs: Optional[int] = 0,
                intervention_consistency_weight: Optional[float] = 0.0,
                intervention_prob_start: Optional[float] = None,
                intervention_anneal_epochs: Optional[int] = 0,
                c_cardinalities: Optional[list] = None,
                ):
        super(Predictor, self).__init__()         
        self.model = model
        self.save_hyperparameters(ignore=["model"], logger=False)

        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs or dict()
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or dict()

        # for regularization
        self.intervention_prob = intervention_prob
        # store the intervention policy
        self.test_interv_policy = test_interv_policy
        self.test_interv_noise = test_interv_noise  
        self.propagator_warmup_epochs = propagator_warmup_epochs
        # KL(softmax(y|c_pred) || softmax(y|c_gt).detach()) coefficient.
        self.intervention_consistency_weight = float(intervention_consistency_weight or 0.0)
        # Linear anneal of training intervention_prob: start -> intervention_prob.
        self.intervention_prob_start = (
            float(intervention_prob_start) if intervention_prob_start is not None else None
        )
        self.intervention_anneal_epochs = int(intervention_anneal_epochs or 0)

        self.c_names = c_names if c_names is not None else []
        self.c_cardinalities = c_cardinalities if c_cardinalities is not None else []
        self.n_concepts = len(self.c_names)

        if metrics is None:
            metrics = dict()
        self._set_metrics(metrics)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    @staticmethod
    def _check_metric(metric):
        metric = metric.clone()
        metric.reset()
        return metric

    @staticmethod
    def _weighted_accuracy_np(targets, preds):
        targets = np.asarray(targets)
        preds = np.asarray(preds)
        if targets.size == 0:
            return 0.0
        classes, counts = np.unique(targets, return_counts=True)
        weights = np.zeros(targets.shape[0], dtype=float)
        for cls, count in zip(classes, counts):
            weights[targets == cls] = 1.0 / max(float(count), 1.0)
        return float(((targets == preds).astype(float) * weights).sum() / max(weights.sum(), 1e-8))

    @staticmethod
    def _build_task_report(targets, preds, class_names):
        from sklearn.metrics import f1_score
        labels = list(range(len(class_names)))
        per_class_f1 = f1_score(targets, preds, labels=labels, average=None, zero_division=0)
        return {
            'macro_f1': float(f1_score(targets, preds, labels=labels, average='macro', zero_division=0)),
            'weighted_accuracy': Predictor._weighted_accuracy_np(targets, preds),
            'per_class_f1': [float(v) for v in per_class_f1],
            'class_names': class_names,
        }

    @staticmethod
    def _print_task_report(title, report):
        print(f"\n=== {title} ===")
        print(f"Macro-F1: {report['macro_f1']:.4f}")
        print(f"Weighted accuracy: {report['weighted_accuracy']:.4f}")
        print("Per-class F1:")
        for name, value in zip(report['class_names'], report['per_class_f1']):
            print(f"  {name}: {value:.4f}")
    
    def _set_metrics(self, metrics):
        # --- accuracy metrics ---
        from src.metrics import MacroF1, MacroPrecision, MacroRecall, WeightedAccuracy
        # --- accuracy metrics ---
        base_acc = metrics.get('classification_acc')
        y_acc_metrics = {
            'y_accuracy': base_acc,
            'y_weighted_accuracy': WeightedAccuracy(),
            'y_f1_macro': MacroF1(),
            'y_precision_macro': MacroPrecision(),
            'y_recall_macro': MacroRecall()
        }
        
        c_acc_metrics = {}
        for ki, k in enumerate(self.c_names):
            card = self.c_cardinalities[ki] if ki < len(self.c_cardinalities) else 2
            if card == 1:
                # Continuous concept: use MAE (regression)
                from src.metrics import RegressionMAE
                c_acc_metrics[k] = RegressionMAE()
                c_acc_metrics[f"{k}_mae"] = RegressionMAE()
            else:
                # Categorical concept: use accuracy + F1
                c_acc_metrics[k] = base_acc
                c_acc_metrics[f"{k}_f1_macro"] = MacroF1()


        # task accuracy metrics
        self.train_y_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in y_acc_metrics.items()},
            prefix="train/y/")
        self.val_y_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in y_acc_metrics.items()},
            prefix="val/y/")
        self.test_y_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in y_acc_metrics.items()},
            prefix="test/y/")
        
        # --- concept accuracy metrics ---
        self.train_c_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in c_acc_metrics.items()},
            prefix="train/c/")
        self.val_c_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in c_acc_metrics.items()},
            prefix="val/c/")
        self.test_c_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in c_acc_metrics.items()},
            prefix="test/c/")      
          
        if self.model.has_concepts:
            # --- ground truth intervention metrics ---
            # Intervention metrics always measure TASK classification accuracy,
            # regardless of concept cardinality
            intervention_acc_metrics = {}
            intervention_acc_metrics['_baseline'] = metrics.get('classification_acc')
            intervention_acc_metrics['_baseline_f1_macro'] = MacroF1()
            for k in self.c_names:
                intervention_acc_metrics[k] = metrics.get('classification_acc')
                intervention_acc_metrics[f"{k}_f1_macro"] = MacroF1()
            
            interv_policy_safe = self.test_interv_policy if self.test_interv_policy is not None else []
            c_acc_levels_metrics = {f'level {n}': metrics.get('classification_acc')
                                    for n in range(0, len(interv_policy_safe)+1)}
            
            # task accuracy after invervention on each individual concept 
            # (one metric for each concept)
            self.test_intervention_single_y = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in intervention_acc_metrics.items()},
                prefix="test_intervention/single/y/")
            
            # task accuracy after intervention of each graph level
            self.test_intervention_level_y = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in c_acc_levels_metrics.items()},
                prefix="test_intervention/level/y/")

            # individual concept accuracy (task ancestors only, according to the policy) after 
            # intervention on levels defined by the policy
            nodes_per_level = {}
            indices_in_policy = list(itertools.chain(*interv_policy_safe))
            c_names_in_policy = [self.c_names[i] for i in indices_in_policy]
            nodes_per_level.update({
                f'level {l}/node {c}': metrics.get('classification_acc')
                for l in range(len(interv_policy_safe) + 1)
                for c in c_names_in_policy
            })
            self.test_intervention_level_c = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in nodes_per_level.items()},
                prefix="test_intervention/level/c/")
            

            # --- fairness metrics ---
            self.cace = MetricCollection(
                metrics = {'before': self._check_metric(metrics.get('cace')),
                           'after': self._check_metric(metrics.get('cace'))},   
                prefix="test_intervention/cace/")

    def log_metrics(self, metrics, **kwargs):
        """"""
        self.log_dict(
            metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True, **kwargs
        )

    def log_loss(self, name, loss, **kwargs):
        """"""
        self.log(
            name + "_loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            **kwargs,
        )

    def _unpack_batch(self, batch):
        """
        Unpack a batch into data and preprocessing dictionaries.
        """
        return batch['x'], batch['c'], batch['y']

    @staticmethod
    def _model_inputs(x, c, intervention_index, batch=None):
        inputs = {'x': x, 'c': c, 'intervention_index': intervention_index}
        if isinstance(batch, dict):
            for optional_key in ('x_baseline', 'x_seg_curr', 'x_seg_base', 'clinical_features'):
                if optional_key in batch:
                    inputs[optional_key] = batch[optional_key]
        return inputs
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        # add batch_size to batch
        if isinstance(batch, dict):
            batch['batch_size'] = batch['x'].shape[0]
        else:
            raise NotImplementedError("Only dict batches are supported")
        return batch

    def _current_intervention_prob(self):
        if (self.intervention_prob_start is None
                or self.intervention_anneal_epochs <= 0):
            return float(self.intervention_prob)
        e = float(self.current_epoch)
        frac = min(1.0, max(0.0, e / float(self.intervention_anneal_epochs)))
        return (self.intervention_prob_start
                + frac * (float(self.intervention_prob) - self.intervention_prob_start))

    def get_intervention_index(self, c_shape, step):
        """
        Get intervention index for training time intervention.
        Args:
            c_shape: shape of the concept tensor
            step: (str) 'train' or 'val'
        """
        if step == 'train':
            p = self._current_intervention_prob()
            intervention_index = torch.bernoulli(torch.ones(c_shape) * p)
        else:
            intervention_index = torch.zeros(c_shape)
        return intervention_index.to("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_intervention(self, batch):
        if self.model.has_concepts:
            x, c, y = self._unpack_batch(batch)
            # maybe add noise
            if self.test_interv_noise > 0:
                x = x + torch.randn_like(x) * self.test_interv_noise

            # baseline task accuracy
            # do not intervene
            intervention_index = get_test_intervention_index(c.shape, [])
            inputs = self._model_inputs(x, c, intervention_index, batch)
            # forward pass with intervention at test time
            y_output, c_output = self.forward(**inputs)
            y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
            # update metric after intervention:
            # how well can we predict y?
            self.test_intervention_single_y['_baseline'].update(y_hat, y)
            self.test_intervention_single_y['_baseline_f1_macro'].update(y_hat, y)

            # interventions on individual concepts
            for i, c_name_i in enumerate(self.c_names):
                if c_name_i in self.model.virtual_roots: continue
                # intervene on concept c_name_i
                intervention_index = get_test_intervention_index(c.shape, i)
                inputs = self._model_inputs(x, c, intervention_index, batch)
                # forward pass with intervention at test time
                y_output, c_output = self.forward(**inputs)
                y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
                # update metric after intervention:
                # after interveening on concept c_name_i, how well can we predict y?
                self.test_intervention_single_y[c_name_i].update(y_hat, y)
                f1_key = f"{c_name_i}_f1_macro"
                if f1_key in self.test_intervention_single_y:
                    self.test_intervention_single_y[f1_key].update(y_hat, y)

            # level intervention
            for l in range(0, len(self.test_interv_policy)+1):
                nodes = list(itertools.chain(*self.test_interv_policy[:l]))
                intervention_index = get_test_intervention_index(c.shape, nodes)
                inputs = self._model_inputs(x, c, intervention_index, batch)
                # forward pass with intervention at test time
                y_output, c_output = self.forward(**inputs)
                y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
                # update metric after intervention:
                # after interveening on a level of the policy, how well can we predict y?
                self.test_intervention_level_y[f'level {l}'].update(y_hat, y)
                # update metric after intervention:
                # after interveening on a level of the policy, how well can we predict each child concept?
                indices_in_policy = list(itertools.chain(*self.test_interv_policy))
                for node_index in indices_in_policy:
                    c_name = self.c_names[node_index]
                    if c_name in c_hat:
                        self.test_intervention_level_c[f'level {l}/node {c_name}'].update(c_hat[c_name], c[:,node_index])
                    else:
                        # if the concept is not in the output, we cannot compute the metric for that concept
                        # this can happen if the model does not predict all concepts
                        pass

    def test_intervention_fairness(self, batch):
        if self.model.has_concepts:
            x, c, y = self._unpack_batch(batch)

            # get a concept pair i,j (node j has to be a bottleneck for node i to the task)
            i = self.c_names.index('Attractive')
            j = self.c_names.index('Qualified')

            # compute the cace before the do-intervention on concept j
            # different do-interventions on concept i, effect on the task
            interv_index, interv_values = get_test_intervention_index(c.shape, i, values=1)
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            y_hat_before_do_1, _ = self.model.filter_output_for_metric(y_output, c_output)
            interv_index, interv_values = get_test_intervention_index(c.shape, i, values=0)
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            y_hat_before_do_0, _ = self.model.filter_output_for_metric(y_output, c_output)
            self.cace['before'].update(y_hat_before_do_1, y_hat_before_do_0)

            # on causal models like causal cem, because of the way they are implemented, is not necessary to strip eedges
            # after interventions, as interventions fix the values of the concept and previous calculations are useless
            # at most there is a little overhead in the forward pass
            # if self.model.is_causal:
            #     self.model.remove_edges(j)

            # compute the cace after the do-intervention on concept j
            # different do-interventions on concept i, effect on the task
            interv_index, interv_values = get_test_intervention_index(c.shape, [j,i], values=[1,1])
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            y_hat_after_do_1, _ = self.model.filter_output_for_metric(y_output, c_output)
            interv_index, interv_values = get_test_intervention_index(c.shape, [j,i], values=[1,0])
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            y_hat_after_do_0, _ = self.model.filter_output_for_metric(y_output, c_output)
            self.cace['after'].update(y_hat_after_do_1, y_hat_after_do_0)

            self.log_metrics(self.cace, batch_size=batch['batch_size'])


    def update_and_log_metrics(self, step, y_hat, y, c_hat, c, batch):
        # update and log task metrics
        y_collection = getattr(self, f"{step}_y_metrics")
        y_collection.update(y_hat, y)
        self.log_metrics(y_collection, batch_size=batch['batch_size'])
        # update and log concept metrics
        c_collection = getattr(self, f"{step}_c_metrics")
        # log metrics for all predicted concepts
        # (the collection contains all concepts, but some models predicts only a subset)
        if c_hat is not None:
            # DIAG: V_NE c_hat presence check (remove after diagnosing)
            if step == "test" and not getattr(self, "_diag_printed", False):
                missing = [n for n in self.c_names if n not in c_hat]
                print(f"[DIAG c_hat MISSING for test]: {missing}")
                print(f"[DIAG c_hat KEYS for test]: {sorted(c_hat.keys())}")
                for k in ('non_enhancing_volume_cm3', 'followup_non_enhancing_volume_cm3',
                         'delta_enhancing_absolute', 'delta_non_enhancing_absolute',
                         'delta_non_enhancing_percent', 'delta_spd_absolute'):
                    if k in c_hat:
                        v = c_hat[k]
                        gt = c[:, self.c_names.index(k)]
                        print(f"[DIAG {k}] pred[:3]={v[:3].squeeze().tolist()}  gt[:3]={gt[:3].tolist()}  shape={tuple(v.shape)}")
                    else:
                        print(f"[DIAG {k}] NOT IN c_hat")
                self._diag_printed = True
            for k, v in c_hat.items():
                c_idx = self.c_names.index(k)
                c_collection[k].update(v, c[:, c_idx])
                # Update secondary metric (F1 for categorical, MAE for continuous)
                f1_key = f"{k}_f1_macro"
                if f1_key in c_collection:
                    c_collection[f1_key].update(v, c[:, c_idx])
                mae_key = f"{k}_mae"
                if mae_key in c_collection:
                    c_collection[mae_key].update(v, c[:, c_idx])
        self.log_metrics(c_collection, batch_size=batch['batch_size'])

    def shared_step(self, batch, step):
        x, c, y = self._unpack_batch(batch)
        intervention_index = self.get_intervention_index(c.shape, step=step)
        inputs = self._model_inputs(x, c, intervention_index, batch)

        run_consistency = (
            step == 'train'
            and self.intervention_consistency_weight > 0
            and self.model.has_concepts
        )
        if run_consistency:
            # Ask the model to also return x_encoded so the second pass can
            # skip the (very expensive) 3D encoder.
            y_output, c_output, x_encoded = self.model(
                **inputs, return_x_encoded=True
            )
        else:
            y_output, c_output = self.forward(**inputs)

        # Compute loss
        y_hat_loss, c_hat_loss = self.model.filter_output_for_loss(y_output, c_output)
        loss = self.model.loss(y_hat_loss, y, c_hat_loss, c)

        if run_consistency:
            # Second forward: all-GT concepts, reuses the cached x_encoded so
            # the 3D MedicalNet backbone runs only once per training step.
            gt_index = torch.ones_like(intervention_index)
            with torch.no_grad():
                y_output_gt, _ = self.model(
                    x=x, c=c, intervention_index=gt_index,
                    cached_x_encoded=x_encoded,
                )
            eps = 1e-6
            log_p = torch.log(y_output.clamp(min=eps, max=1.0 - eps))
            q = y_output_gt.clamp(min=eps, max=1.0 - eps).detach()
            kl = torch.nn.functional.kl_div(log_p, q, reduction='batchmean')
            loss = loss + self.intervention_consistency_weight * kl
            self.log_loss("train/consistency", kl, batch_size=batch['batch_size'])
        return loss, y_output, c_output, y, c

    def training_step(self, batch, batch_idx):
        loss, y_output, c_output, y, c = self.shared_step(batch, step='train')
        if torch.isnan(loss).any():
            print(f'at epoc: {self.current_epoch}, batch: {batch_idx}')
            print('Loss has nan')
        # Update metrics and log
        y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
        self.update_and_log_metrics("train", y_hat, y, c_hat, c, batch)
        self.log_loss("train", loss, batch_size=batch['batch_size'])
        return loss
    
    def on_train_epoch_start(self):
        """Staged training: freeze propagators during warmup, then unfreeze."""
        if self.propagator_warmup_epochs > 0 and hasattr(self.model, 'propagators'):
            in_warmup = self.current_epoch < self.propagator_warmup_epochs
            for param in self.model.propagators.parameters():
                param.requires_grad = not in_warmup
            if self.current_epoch == self.propagator_warmup_epochs:
                print(f"\n>>> Warmup complete (epoch {self.current_epoch}): "
                      f"unfreezing propagators for joint training <<<\n")
            elif self.current_epoch == 0:
                print(f"\n>>> Staged training: freezing propagators for "
                      f"{self.propagator_warmup_epochs} warmup epochs <<<\n")

    def on_train_epoch_end(self):
        # Set the current epoch for SCBM and update the list of concept probs for computing the concept percentiles
        if type(self.model).__name__ == 'SCBM':
            self.model.training_epoch = self.current_epoch
            # self.model.concept_pred = torch.cat(self.model.concept_pred_tmp, dim=0) 
            # self.model.concept_pred_tmp = []        

    def validation_step(self, batch, batch_idx):
        val_loss, y_output, c_output, y, c = self.shared_step(batch, step='val')
        # Update metrics and log
        y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
        self.update_and_log_metrics("val", y_hat, y, c_hat, c, batch)
        self.log_loss("val", val_loss, batch_size=batch['batch_size'])
        return val_loss
    
    def test_step(self, batch, batch_idx):
        test_loss, y_output, c_output, y, c = self.shared_step(batch, step='test')
        # Update metrics and log
        y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
        self.update_and_log_metrics("test", y_hat, y, c_hat, c, batch)
        self.log_loss("test", test_loss, batch_size=batch['batch_size'])
        # test-time interventions
        self.test_intervention(batch)
        if 'Qualified' in self.c_names:
            self.test_intervention_fairness(batch)
        # accumulate for threshold-optimised prediction and CaCE
        if not hasattr(self, '_cm_probs'):
            self._cm_probs, self._cm_targets = [], []
            self._cm_concepts = []
        self._cm_probs.append(y_hat.detach().cpu())
        self._cm_targets.append(y.flatten().long().cpu())
        self._cm_concepts.append(c.detach().cpu())
        return test_loss

    def on_test_epoch_end(self):
        if hasattr(self, '_cm_probs') and self._cm_probs:
            all_probs = torch.cat(self._cm_probs)
            all_targets = torch.cat(self._cm_targets)
            n_classes = all_probs.size(1)
            if n_classes == 2:
                class_names = ['Non-PD(0)', 'PD(1)']
            else:
                class_names = ['CR(0)', 'PR(1)', 'SD(2)', 'PD(3)']

            # Standard argmax confusion matrix
            all_preds = all_probs.argmax(dim=1)
            cm = np.zeros((n_classes, n_classes), dtype=int)
            for t, p in zip(all_targets.numpy(), all_preds.numpy()):
                cm[t][p] += 1
            print('\n=== Confusion Matrix [argmax] (rows=true, cols=pred) ===')
            header = '       ' + '  '.join(f'{c:6s}' for c in class_names)
            print(header)
            for i, row in enumerate(cm):
                print(f'{class_names[i]:6s}  ' + '  '.join(f'{v:6d}' for v in row))
            print('Per-class recall:')
            for i in range(n_classes):
                total = cm[i].sum()
                recall = cm[i][i] / total if total > 0 else 0.
                print(f'  {class_names[i]}: {recall:.3f}  ({cm[i][i]}/{total})')
            pickle.dump({'confusion_matrix': cm.tolist(), 'class_names': class_names},
                        open('results/confusion_matrix.pkl', 'wb'))

            targets_np = all_targets.numpy()
            preds_np = all_preds.numpy()
            self.latest_task_reports = {}
            if n_classes == 4:
                four_class_report = self._build_task_report(
                    targets_np, preds_np, ['CR', 'PR', 'SD', 'PD'])
                binary_targets = (targets_np == 3).astype(int)
                binary_preds = (preds_np == 3).astype(int)
                binary_report = self._build_task_report(
                    binary_targets, binary_preds, ['Non-PD', 'PD'])
                self.latest_task_reports['four_class'] = four_class_report
                self.latest_task_reports['binary'] = binary_report
                self._print_task_report('4-class task metrics [argmax]', four_class_report)
                self._print_task_report('Binary task metrics [4-class argmax remapped: CR/PR/SD=Non-PD, PD=PD]',
                                        binary_report)
            elif n_classes == 2:
                binary_report = self._build_task_report(targets_np, preds_np, ['Non-PD', 'PD'])
                self.latest_task_reports['binary'] = binary_report
                self._print_task_report('Binary task metrics [argmax]', binary_report)
            pickle.dump(self.latest_task_reports, open('results/task_reports.pkl', 'wb'))

            # Threshold-optimised prediction (grid search on val, apply to test)
            from sklearn.metrics import f1_score as _f1

            # Try engine-stored ref first (reliable), then PL trainer attribute
            val_dl = getattr(self, '_val_dataloader_ref', None)
            if val_dl is None:
                val_dl = getattr(self.trainer, 'val_dataloaders', None)
            if val_dl is None:
                print('  [WARNING] val_dataloaders not available, falling back to test set')
                val_probs = all_probs
                val_targets_np = all_targets.numpy()
            else:
                if isinstance(val_dl, list):
                    val_dl = val_dl[0]
                val_probs_list, val_targets_list = [], []
                self.model.eval()
                with torch.no_grad():
                    for vbatch in val_dl:
                        vbatch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                  for k, v in vbatch.items()}
                        vbatch['batch_size'] = vbatch['x'].shape[0]
                        vx, vc, vy = self._unpack_batch(vbatch)
                        v_interv = get_test_intervention_index(vc.shape, [])
                        vy_out, vc_out = self.forward(**self._model_inputs(vx, vc, v_interv, vbatch))
                        vy_hat, _ = self.model.filter_output_for_metric(vy_out, vc_out)
                        val_probs_list.append(vy_hat.detach().cpu())
                        val_targets_list.append(vy.flatten().long().cpu())
                val_probs = torch.cat(val_probs_list)
                val_targets_np = torch.cat(val_targets_list).numpy()
                print(f'  [threshold search] val set size: {len(val_targets_np)}')

            best_f1, best_boosts = -1, tuple([1.0] * n_classes)
            if n_classes == 2:
                # Binary: boost minority class (Non-PD = class 0)
                for minority_boost in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
                    adjusted = val_probs.clone()
                    adjusted[:, 0] *= minority_boost
                    preds_adj = adjusted.argmax(dim=1).numpy()
                    f1_val = _f1(val_targets_np, preds_adj, average='macro', zero_division=0)
                    if f1_val > best_f1:
                        best_f1, best_boosts = f1_val, (minority_boost, 1.0)
                boost_label = f'Non-PD×{best_boosts[0]}'
            else:
                for cr_boost in [1.0, 1.5, 2.0, 2.5, 3.0]:
                    for pr_boost in [1.0, 1.5, 2.0, 2.5, 3.0]:
                        adjusted = val_probs.clone()
                        adjusted[:, 0] *= cr_boost
                        adjusted[:, 1] *= pr_boost
                        preds_adj = adjusted.argmax(dim=1).numpy()
                        f1_val = _f1(val_targets_np, preds_adj, average='macro', zero_division=0)
                        if f1_val > best_f1:
                            best_f1, best_boosts = f1_val, (cr_boost, pr_boost)
                boost_label = f'CR×{best_boosts[0]}, PR×{best_boosts[1]}'
            print(f'  [threshold search on VAL set] best val macro-F1={best_f1:.4f}  boosts={boost_label}')

            # Apply best boosts to test predictions
            adjusted = all_probs.clone()
            for bi in range(min(len(best_boosts), n_classes)):
                adjusted[:, bi] *= best_boosts[bi]
            thresh_preds = adjusted.argmax(dim=1)
            best_f1_test = _f1(targets_np, thresh_preds.numpy(), average='macro', zero_division=0)
            cm_t = np.zeros((n_classes, n_classes), dtype=int)
            for t, p in zip(targets_np, thresh_preds.numpy()):
                cm_t[t][p] += 1
            print(f'\n=== Confusion Matrix [threshold-opt, {boost_label}] ===')
            print(f'    Val macro-F1: {best_f1:.4f}  |  TEST macro-F1: {best_f1_test:.4f}')
            header = '       ' + '  '.join(f'{c:6s}' for c in class_names)
            print(header)
            for i, row in enumerate(cm_t):
                print(f'{class_names[i]:6s}  ' + '  '.join(f'{v:6d}' for v in row))
            print('Per-class recall:')
            for i in range(n_classes):
                total = cm_t[i].sum()
                recall = cm_t[i][i] / total if total > 0 else 0.
                print(f'  {class_names[i]}: {recall:.3f}  ({cm_t[i][i]}/{total})')
            pickle.dump({'confusion_matrix_threshold': cm_t.tolist(),
                         'class_names': class_names,
                         'best_boosts': best_boosts,
                         'val_macro_f1': best_f1,
                         'threshold_macro_f1': best_f1_test},
                        open('results/confusion_matrix_threshold.pkl', 'wb'))
            self._cm_probs, self._cm_targets = [], []

        # baseline task metrics
        y_baseline = self.test_y_metrics['y_accuracy'].compute().item()
        y_f1_macro = self.test_y_metrics['y_f1_macro'].compute().item()
        y_weighted_accuracy = self.test_y_metrics['y_weighted_accuracy'].compute().item()
        self.latest_test_metrics = dict(getattr(self, 'latest_task_reports', {}))
        self.latest_test_metrics.update({
            'y_accuracy': y_baseline,
            'y_f1_macro': y_f1_macro,
            'y_weighted_accuracy': y_weighted_accuracy,
        })
        print(f"Baseline task accuracy: {y_baseline}")
        print(f"Baseline task macro-F1: {y_f1_macro}")
        print(f"Baseline task weighted accuracy: {y_weighted_accuracy}")
        pickle.dump({'_baseline': y_baseline,
                     '_macro_f1': y_f1_macro,
                     '_weighted_accuracy': y_weighted_accuracy},
                    open(f'results/y_accuracy.pkl', 'wb'))

        # baseline concept accuracy
        c_baseline = {}
        for k, metric in self.test_c_metrics.items():
            k = _remove_prefix(k, self.test_c_metrics.prefix)
            c_baseline[k] = metric.compute().item()
            print(f"Baseline concept accuracy for {k}: {c_baseline[k]}")
        pickle.dump(c_baseline, open(f'results/c_accuracy.pkl', 'wb'))

        if self.model.has_concepts:
            # task accuracy after invervention on each individual concept
            y_int = {}
            for k, metric in self.test_intervention_single_y.items():
                c_name = _remove_prefix(k, self.test_intervention_single_y.prefix)
                y_int[c_name] = metric.compute().item()
                metric_label = "macro-F1" if c_name.endswith("_f1_macro") else "accuracy"
                print(f"Task {metric_label} after intervention on {c_name}: {y_int[c_name]}")

            baseline_f1 = y_int.get('_baseline_f1_macro')
            if baseline_f1 is not None:
                for c_name in self.c_names:
                    if c_name in self.model.virtual_roots:
                        continue
                    f1_key = f"{c_name}_f1_macro"
                    if f1_key in y_int:
                        y_int[f"{c_name}_baseline_f1_macro"] = baseline_f1
                        y_int[f"{c_name}_delta_f1_macro"] = y_int[f1_key] - baseline_f1
            pickle.dump(y_int, open(f'results/single_c_interventions_on_y.pkl', 'wb'))

            # task accuracy after intervention of each policy level
            y_int = {}
            for k, metric in self.test_intervention_level_y.items():
                level = _remove_prefix(k, self.test_intervention_level_y.prefix)
                y_int[level] = metric.compute().item()
                print(f"Task accuracy after intervention on {level}: {y_int[level]}")
            pickle.dump(y_int, open(f'results/level_interventions_on_y.pkl', 'wb'))

            # individual concept accuracy after intervention of each policy level
            c_int = {}
            for k, metric in self.test_intervention_level_c.items():
                level = _remove_prefix(k, self.test_intervention_level_c.prefix)
                c_int[level] = metric.compute().item()
                print(f"Concept accuracy after intervention on {level}: {c_int[level]}")
            pickle.dump(c_int, open(f'results/level_interventions_on_c.pkl', 'wb'))

            # save graph and concepts
            pickle.dump({'concepts':self.c_names,
                         'policy':self.test_interv_policy}, open("graph.pkl", 'wb'))
            
            pickle.dump({'policy':self.test_interv_policy}, open("policy.pkl", 'wb'))

            # Per-concept CaCE
            self._compute_per_concept_cace()

    def _compute_per_concept_cace(self):
        """Compute CaCE (Total Variation Distance) for each concept on the task."""
        try:
            test_dl = getattr(self.trainer, 'test_dataloaders', None)
            if test_dl is None:
                print("  [CaCE] test_dataloaders not available, skipping CaCE computation")
                return
            if isinstance(test_dl, list):
                test_dl = test_dl[0]

            # Determine concept cardinalities
            if hasattr(self.model, 'combo_info'):
                cardinalities = self.model.combo_info['cardinality'][:self.n_concepts]
            elif hasattr(self.model, 'c_info'):
                cardinalities = self.model.c_info['cardinality']
            else:
                # Infer from accumulated concept data
                all_c = torch.cat(self._cm_concepts) if hasattr(self, '_cm_concepts') and self._cm_concepts else None
                if all_c is None:
                    print("  [CaCE] No concept data accumulated, skipping"); return
                cardinalities = []
                for ci in range(all_c.shape[1]):
                    vals = all_c[:, ci]
                    unique = vals.unique()
                    cardinalities.append(2 if len(unique) == 2 else 1)

            all_c = torch.cat(self._cm_concepts) if hasattr(self, '_cm_concepts') and self._cm_concepts else None
            if all_c is None:
                print("  [CaCE] No concept data, skipping"); return

            high_vals = []
            low_vals = []
            for ci in range(self.n_concepts):
                card = cardinalities[ci]
                if card == 1:  # continuous: ±1σ from mean (z-scored → +1 / -1)
                    high_vals.append(1.0)
                    low_vals.append(-1.0)
                elif card == 2:  # binary
                    high_vals.append(1.0)
                    low_vals.append(0.0)
                else:  # multi-class (e.g. 3-class RANO): sweep full range
                    high_vals.append(float(card - 1))
                    low_vals.append(0.0)

            print(f"\n=== Per-Concept CaCE (Concept Causal Effect) ===")
            cace_results = {}

            self.model.eval()
            device = next(self.model.parameters()).device

            for ci, c_name in enumerate(self.c_names):
                if c_name in getattr(self.model, 'virtual_roots', []):
                    continue

                # Accumulate do-intervention probabilities
                probs_high_all, probs_low_all = [], []
                with torch.no_grad():
                    for batch in test_dl:
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()}
                        x, c, y = self._unpack_batch(batch)
                        B = c.shape[0]

                        interv_idx = torch.zeros_like(c)
                        interv_idx[:, ci] = 1.0

                        # do(c_i = high)
                        c_high = c.clone()
                        c_high[:, ci] = high_vals[ci]
                        y_out_h, c_out_h = self.forward(**self._model_inputs(x, c_high, interv_idx, batch))
                        y_hat_h, _ = self.model.filter_output_for_metric(y_out_h, c_out_h)
                        probs_high_all.append(y_hat_h.detach().cpu())

                        # do(c_i = low)
                        c_low = c.clone()
                        c_low[:, ci] = low_vals[ci]
                        y_out_l, c_out_l = self.forward(**self._model_inputs(x, c_low, interv_idx, batch))
                        y_hat_l, _ = self.model.filter_output_for_metric(y_out_l, c_out_l)
                        probs_low_all.append(y_hat_l.detach().cpu())

                probs_high = torch.cat(probs_high_all, dim=0)
                probs_low = torch.cat(probs_low_all, dim=0)

                avg_high = probs_high.mean(dim=0)
                avg_low = probs_low.mean(dim=0)

                tv_distance = 0.5 * (avg_high - avg_low).abs().sum().item()
                per_class = (avg_high - avg_low).tolist()

                cace_results[c_name] = {
                    'cace_tv': tv_distance,
                    'per_class': per_class,
                    'high_val': high_vals[ci],
                    'low_val': low_vals[ci],
                    'cardinality': cardinalities[ci],
                    'avg_probs_high': avg_high.tolist(),
                    'avg_probs_low': avg_low.tolist(),
                }
                print(f"  {c_name}: CaCE(TV)={tv_distance:.4f}  "
                      f"high={high_vals[ci]:.2f} low={low_vals[ci]:.2f}  "
                      f"per_class={[f'{v:.4f}' for v in per_class]}")

            # Sort by CaCE magnitude
            sorted_concepts = sorted(cace_results.items(), key=lambda x: x[1]['cace_tv'], reverse=True)
            print(f"\n  CaCE Ranking (descending):")
            for rank, (name, info) in enumerate(sorted_concepts, 1):
                print(f"    {rank}. {name}: {info['cace_tv']:.4f}")

            pickle.dump(cace_results, open('results/cace_per_concept.pkl', 'wb'))
            print(f"  CaCE results saved to results/cace_per_concept.pkl")

        except Exception as e:
            print(f"  [CaCE] Error computing CaCE: {e}")
            import traceback; traceback.print_exc()
        finally:
            # Clean up accumulated concepts
            self._cm_concepts = []

    def configure_optimizers(self):
        """"""
        cfg = dict()
        optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        cfg["optimizer"] = optimizer
        if self.scheduler_class is not None:
            metric = self.scheduler_kwargs.pop("monitor", None)
            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
            cfg["lr_scheduler"] = scheduler
            if metric is not None:
                cfg["monitor"] = metric
        return cfg
