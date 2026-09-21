"""Diff2Obs Defense Implementation for Gradient Inversion Protection in Federated Learning.

Implements the official Diff2Obs equations:
  - Obs-I (Obfuscation-Interpolation)
  - Obs-M (Obfuscation-Masking)
  - Obs-S (Obfuscation-Statistic Match)
"""

import torch
import torch.nn as nn


class Diff2ObsDefense:
    def __init__(self, rho_min=0.4, total_rounds=2000, num_candidates=10, strategy='Obs-I'):
        self.rho_min = rho_min
        self.total_rounds = total_rounds
        self.num_candidates = num_candidates
        self.strategy = strategy

    def get_decay_factor(self, round_r):
        """Calculate dynamic decay weight rho(r) in range [rho_min, 1.0]."""
        r = min(max(round_r, 0), self.total_rounds)
        decay = (1.0 - self.rho_min) * (r / max(self.total_rounds, 1))
        return 1.0 - decay

    def select_label_agnostic(self, model, ground_truth, labels):
        """Select a synthetic target label distinct from ground truth."""
        was_training = model.training
        model.eval()
        with torch.no_grad():
            outputs = model(ground_truth)
            num_classes = outputs.shape[-1]
            true_label = labels[0].item() if labels.dim() > 0 else labels.item()
            candidate_labels = [c for c in range(num_classes) if c != true_label]
            synth_label = candidate_labels[torch.randint(0, len(candidate_labels), (1,)).item()] if candidate_labels else true_label
        if was_training:
            model.train()
        return synth_label

    def select_synthesized_image(self, model, shape, device='cuda'):
        return torch.randn(shape, device=device)

    def obfuscate_gradients(self, grad_private, grad_synth, round_r=0, strategy=None, mask_prob=None):
        if grad_synth is None:
            return grad_private

        rho = self.get_decay_factor(round_r)
        mode = strategy if strategy is not None else self.strategy

        if mode in ['Obs-I', 'interpolation']:
            # Equation (2): (1 - lambda) * g_l + lambda * g_s
            lam = 1.0 - rho
            return (1.0 - lam) * grad_private + lam * grad_synth

        elif mode in ['Obs-M', 'masking']:
            # Equation (3): (1 - M) * g_l + M * g_s using Bernoulli binary mask M
            p = mask_prob if mask_prob is not None else (1.0 - rho)
            M = torch.bernoulli(torch.full_like(grad_private, fill_value=p))
            return (1.0 - M) * grad_private + M * grad_synth

        elif mode in ['Obs-S', 'statistic_match']:
            # Equation (5): Statistical matching of mean/std + linear combination
            gamma = 1.0 - rho
            eps = 1e-8

            mu_Gl = grad_private.mean()
            sigma_Gl = grad_private.std()

            mu_Gs = grad_synth.mean()
            sigma_Gs = grad_synth.std()

            # Normalize private gradient statistics to match synthetic gradient
            grad_hat = mu_Gs + (sigma_Gs / (sigma_Gl + eps)) * (grad_private - mu_Gl)
            return (1.0 - gamma) * grad_private + gamma * grad_hat

        else:
            lam = 1.0 - rho
            return (1.0 - lam) * grad_private + lam * grad_synth