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
        dist_type='beta',
        concentration=10.0,
    ):
        """Diff2Obs Defense Implementation with Component-Wise Weighting.

        Args:
            rho_min (float): Minimum decay weight floor.
            total_rounds (int): Total FL training rounds.
            num_candidates (int): Number of synthetic candidates generated.
            strategy (str): Defense mode ('Obs-I', 'Obs-M', or 'Obs-S').
            dist_type (str): 'beta' for continuous component-wise mixing or 'bernoulli' for binary selection.
            concentration (float): Concentration parameter S = (alpha + beta) for the Beta distribution.
        """
        self.rho_min = rho_min
        self.total_rounds = total_rounds
        self.num_candidates = num_candidates
        self.strategy = strategy
        self.dist_type = dist_type.lower()
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
        self, shape, target_mean, dist_type=None, device='cuda', dtype=torch.float32
    ):
        """Generates component-wise weights (lambda/gamma/mask) matching gradient tensor dimensions."""
        mode = dist_type if dist_type is not None else self.dist_type

        # Clamp mean to stay strictly within (0, 1) for stable Beta sampling
        p = float(torch.clamp(torch.tensor(target_mean), 1e-5, 1.0 - 1e-5))

        if mode == 'beta':
            alpha = p * self.concentration
            beta = (1.0 - p) * self.concentration
            alpha_tensor = torch.full(shape, alpha, device=device, dtype=dtype)
            beta_tensor = torch.full(shape, beta, device=device, dtype=dtype)
            return Beta(alpha_tensor, beta_tensor).sample()

        elif mode == 'bernoulli':
            return torch.bernoulli(
                torch.full(shape, fill_value=p, device=device, dtype=dtype)
            )

        else:
            raise ValueError(f"Unsupported distribution type '{mode}'. Choose 'beta' or 'bernoulli'.")

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
        target_mean = 1.0 - rho  # Blending factor intensity
        mode = strategy if strategy is not None else self.strategy

        # Component-wise weight tensor lambda/gamma/M ~ Dist(mean = 1 - rho)
        weights = self._sample_weights(
            shape=grad_private.shape,
            target_mean=target_mean,
            dist_type=dist_type,
            device=grad_private.device,
            dtype=grad_private.dtype,
        )

        if mode in ['Obs-I', 'interpolation']:
            # Component-wise linear combination: (1 - Lambda) * g_l + Lambda * g_s
            return (1.0 - weights) * grad_private + weights * grad_synth

        elif mode in ['Obs-M', 'masking']:
            # Component-wise masking: (1 - M) * g_l + M * g_s
            return (1.0 - weights) * grad_private + weights * grad_synth

        elif mode in ['Obs-S', 'statistic_match']:
            # Component-wise statistic matching linear blend
            eps = 1e-8
            mu_Gl = grad_private.mean()
            sigma_Gl = grad_private.std()

            mu_Gs = grad_synth.mean()
            sigma_Gs = grad_synth.std()

            # Normalize private gradient statistics to match synthetic gradient
            grad_hat = mu_Gs + (sigma_Gs / (sigma_Gl + eps)) * (grad_private - mu_Gl)
            return (1.0 - weights) * grad_private + weights * grad_hat

        else:
            return (1.0 - weights) * grad_private + weights * grad_synth