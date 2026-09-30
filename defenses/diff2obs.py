import torch
import torch.nn as nn
from torch.distributions import Beta


class Diff2ObsDefense:
    def __init__(
        self,
        rho_min=0.4,
        total_rounds=2000,
        num_candidates=10,
        strategy='Obs-I',
        dist_type=None,
        concentration=10.0,
    ):
        """Diff2Obs Defense Implementation with Component-Wise Weighting.

        Default Strategy to Distribution Mapping:
          - Obs-I (Interpolation): Beta distribution
          - Obs-S (Statistic Match): Beta distribution
          - Obs-M (Masking): Bernoulli distribution
        """
        self.rho_min = rho_min
        self.total_rounds = total_rounds
        self.num_candidates = num_candidates
        self.strategy = strategy
        self.dist_type = dist_type.lower() if dist_type else None
        self.concentration = concentration

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
            synth_label = (
                candidate_labels[torch.randint(0, len(candidate_labels), (1,)).item()]
                if candidate_labels
                else true_label
            )
        if was_training:
            model.train()
        return synth_label

    def select_synthesized_image(self, model, shape, device='cuda'):
        return torch.randn(shape, device=device)

    def _sample_weights(
        self, shape, target_mean, dist_type, device='cuda', dtype=torch.float32
    ):
        """Generates component-wise weights matching the private gradient shape."""
        p = float(torch.clamp(torch.tensor(target_mean), 1e-5, 1.0 - 1e-5))

        if dist_type == 'beta':
            alpha = p * self.concentration
            beta = (1.0 - p) * self.concentration
            alpha_tensor = torch.full(shape, alpha, device=device, dtype=dtype)
            beta_tensor = torch.full(shape, beta, device=device, dtype=dtype)
            return Beta(alpha_tensor, beta_tensor).sample()

        elif dist_type == 'bernoulli':
            return torch.bernoulli(
                torch.full(shape, fill_value=p, device=device, dtype=dtype)
            )

        else:
            raise ValueError(f"Unsupported distribution type '{dist_type}'. Choose 'beta' or 'bernoulli'.")

    def obfuscate_gradients(
        self,
        grad_private,
        grad_synth,
        round_r=0,
        strategy=None,
        dist_type=None,
    ):
        if grad_synth is None:
            return grad_private

        rho = self.get_decay_factor(round_r)
        target_mean = 1.0 - rho  # Expected intensity (1 - rho)
        mode = strategy if strategy is not None else self.strategy

        # Determine effective distribution (Obs-I & Obs-S -> Beta, Obs-M -> Bernoulli)
        effective_dist = dist_type or self.dist_type
        if effective_dist is None:
            if mode in ['Obs-M', 'masking']:
                effective_dist = 'bernoulli'
            else:  # 'Obs-I', 'Obs-S'
                effective_dist = 'beta'

        # Component-wise weight sampling matching grad_private.shape
        weights = self._sample_weights(
            shape=grad_private.shape,
            target_mean=target_mean,
            dist_type=effective_dist,
            device=grad_private.device,
            dtype=grad_private.dtype,
        )

        if mode in ['Obs-I', 'interpolation']:
            return (1.0 - weights) * grad_private + weights * grad_synth

        elif mode in ['Obs-M', 'masking']:
            return (1.0 - weights) * grad_private + weights * grad_synth

        elif mode in ['Obs-S', 'statistic_match']:
            eps = 1e-8
            mu_Gl = grad_private.mean()
            sigma_Gl = grad_private.std()

            mu_Gs = grad_synth.mean()
            sigma_Gs = grad_synth.std()

            grad_hat = mu_Gs + (sigma_Gs / (sigma_Gl + eps)) * (grad_private - mu_Gl)
            return (1.0 - weights) * grad_private + weights * grad_hat

        else:
            return (1.0 - weights) * grad_private + weights * grad_synth