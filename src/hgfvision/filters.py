"""Diagonal EKF and causal Fisher-scoring uncertainty controllers.

The hierarchical controller is an HGF-inspired innovation-filter extension, not
an implementation of the generalized HGF paper's exact node equations. See
METHOD.md for the generative assumptions, approximations, and causal ordering.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class FilterConfig:
    """All variances are per logical minibatch update, not per image."""

    initial_p: float = 0.1
    initial_q: float = 1e-4
    initial_r: float = 0.05
    ema_rate: float = 0.05
    hyper_variance: float = 1.0
    hyper_drift: float = 0.02
    parent_variance: float = 1.0
    parent_drift: float = 0.001
    q_min: float = 1e-10
    q_max: float = 1.0
    r_min: float = 1e-5
    r_max: float = 10.0
    drift_min: float = 1e-6
    drift_max: float = 1.0
    p_floor: float = 1e-10
    p_ceiling: float = 100.0

    def __post_init__(self) -> None:
        """Reject nonfinite values and invalid bounds before allocating state."""
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for lo, initial, hi in [
            (self.q_min, self.initial_q, self.q_max),
            (self.r_min, self.initial_r, self.r_max),
            (self.drift_min, self.hyper_drift, self.drift_max),
            (self.p_floor, self.initial_p, self.p_ceiling),
        ]:
            if not lo <= initial <= hi or lo >= hi:
                raise ValueError("Initial variance must lie inside ordered bounds")
        if self.ema_rate > 1:
            raise ValueError("ema_rate must be <= 1")


def symmetric(x: Tensor) -> Tensor:
    """Remove small floating-point asymmetries."""
    return (x + x.mT) * 0.5


def cholesky(matrix: Tensor) -> tuple[Tensor, float]:
    """Factor an SPD matrix, reporting any numerical diagonal regularization."""
    matrix = symmetric(matrix)
    if not torch.isfinite(matrix).all():
        raise FloatingPointError("Nonfinite covariance")
    scale = max(float(matrix.diagonal().abs().mean()), 1e-12)
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    for relative in (0.0, 1e-7, 1e-6, 1e-5, 1e-4):
        jitter = relative * scale
        factor, info = torch.linalg.cholesky_ex(matrix + jitter * identity)
        if int(info.max()) == 0:
            return factor, jitter
    raise FloatingPointError("Covariance is not positive definite after bounded jitter")


def variance_score_fisher(
    residual: Tensor, covariance: Tensor, derivatives: Tensor
) -> tuple[Tensor, Tensor, float]:
    """Return Gaussian covariance score and expected Fisher for log variances."""
    factor, jitter = cholesky(covariance)
    inv = torch.cholesky_inverse(factor)
    solved = torch.cholesky_solve(residual[:, None], factor).squeeze(1)
    score = 0.5 * (
        torch.einsum("i,kij,j->k", solved, derivatives, solved)
        - torch.einsum("ij,kji->k", inv, derivatives)
    )
    products = torch.einsum("ij,kjl->kil", inv, derivatives)
    fisher = 0.5 * torch.einsum("kij,lji->kl", products, products)
    return score, symmetric(fisher), jitter


def gaussian_score_update(
    mean: Tensor, covariance: Tensor, score: Tensor, fisher: Tensor
) -> tuple[Tensor, Tensor, float]:
    """Take one Gaussian posterior update under a local quadratic likelihood."""
    prior_factor, j1 = cholesky(covariance)
    precision = torch.cholesky_inverse(prior_factor) + symmetric(fisher)
    post_factor, j2 = cholesky(precision)
    post_cov = symmetric(torch.cholesky_inverse(post_factor))
    return mean + post_cov @ score, post_cov, max(j1, j2)


def parent_score_fisher(
    prior_cov: Tensor, post_cov: Tensor, displacement: Tensor, drift: Tensor
) -> tuple[Tensor, Tensor]:
    """Differentiate the lower Gaussian pseudo-observation's marginal evidence.

    J=B-BAB is the information reaching the parent, where B=Cprior^-1 and
    A=Cpost. This gives zero information when the lower likelihood is flat.
    """
    factor, jitter = cholesky(prior_cov)
    if jitter != 0:
        raise FloatingPointError("Parent covariance unexpectedly required jitter")
    inverse = torch.cholesky_inverse(factor)
    information = symmetric(inverse - inverse @ post_cov @ inverse)
    transformed = inverse @ displacement
    score = 0.5 * drift * (transformed.square() - information.diagonal())
    fisher = 0.5 * torch.outer(drift, drift) * information.square()
    return score, symmetric(fisher)


def fisher_diagnostics(fisher: Tensor) -> dict[str, float]:
    """Report weak or confounded covariance information without altering inference."""
    matrix = symmetric(fisher.detach().to(device="cpu", dtype=torch.float64))
    scale = torch.outer(matrix.diagonal().clamp_min(0), matrix.diagonal().clamp_min(0)).sqrt()
    correlation = torch.where(scale > 0, matrix / scale.clamp_min(torch.finfo(matrix.dtype).tiny), 0)
    correlation.fill_diagonal_(0)
    return {
        "fisher_min_eigenvalue": float(torch.linalg.eigvalsh(matrix).min()),
        "fisher_max_offdiagonal_correlation": float(correlation.abs().max()),
    }


def condition_diagonal(
    residual: Tensor, jacobian: Tensor, prior: Tensor, observation_variance: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, float]:
    """Condition a diagonal prior using one internally consistent numeric dtype."""
    weighted = jacobian * prior
    identity = torch.eye(residual.numel(), device=residual.device, dtype=residual.dtype)
    innovation_cov = weighted @ jacobian.mT + observation_variance * identity
    factor, jitter = cholesky(innovation_cov)
    gain = torch.cholesky_solve(weighted, factor).mT
    displacement = gain @ residual
    posterior = prior - (gain.mT * weighted).sum(0)
    return displacement, posterior, innovation_cov, factor, jitter


class UncertaintyController:
    """Use present innovations only to set the NEXT update's Q and R."""

    def __init__(self, method: str, groups: int, config: FilterConfig, device: torch.device):
        """Initialize shared hyperpriors across adaptive and hierarchical arms."""
        if method not in {"zero_q", "fixed_q", "adaptive_qr", "hierarchical_qr"}:
            raise ValueError(f"Unknown filter method: {method}")
        if groups < 1:
            raise ValueError("At least one parameter group is required")
        self.method, self.config, self.groups = method, config, groups
        self.mean = torch.tensor(
            [math.log(config.initial_q)] * groups + [math.log(config.initial_r)],
            dtype=torch.float32, device=device,
        )
        self.cov = torch.eye(groups + 1, device=device) * config.hyper_variance
        self.parent_mean = torch.full_like(self.mean, math.log(config.hyper_drift))
        self.parent_cov = torch.eye(groups + 1, device=device) * config.parent_variance
        self.last_diagnostics: dict[str, Any] = {}

    def prediction(self) -> tuple[Tensor, Tensor]:
        """Read causal variance predictions without incorporating current labels."""
        q = self.mean[:-1].exp()
        if self.method == "zero_q":
            q = torch.zeros_like(q)
        return q, self.mean[-1].exp()

    @torch.no_grad()
    def observe(self, residual: Tensor, covariance: Tensor, derivatives: Tensor) -> None:
        """Update hyperstates after the parameter update has been computed."""
        cfg = self.config
        if self.method in {"zero_q", "fixed_q"}:
            r = (1 - cfg.ema_rate) * self.mean[-1].exp()
            r = r + cfg.ema_rate * residual.square().mean().to(self.mean)
            self.mean[-1] = r.clamp(cfg.r_min, cfg.r_max).log()
            self.last_diagnostics = {"hyper_bound_hits": 0, "parent_bound_hits": 0, "hyper_jitter": 0.0}
            return
        drift = self.parent_mean.exp() if self.method == "hierarchical_qr" else torch.full_like(self.mean, cfg.hyper_drift)
        prior_cov = self.cov + torch.diag(drift)
        old_mean = self.mean.clone()
        score, fisher, j1 = variance_score_fisher(residual, covariance, derivatives)
        information_diagnostics = fisher_diagnostics(fisher)
        score, fisher = score.to(old_mean), fisher.to(old_mean)
        updated_mean, updated_cov, j2 = gaussian_score_update(old_mean, prior_cov, score, fisher)
        parent_hits = 0
        if self.method == "hierarchical_qr":
            upper_score, upper_fisher = parent_score_fisher(prior_cov, updated_cov, updated_mean - old_mean, drift)
            upper_prior = self.parent_cov + torch.eye(len(old_mean), device=old_mean.device) * cfg.parent_drift
            upper_mean, upper_cov, j3 = gaussian_score_update(self.parent_mean, upper_prior, upper_score, upper_fisher)
            bounded_upper = upper_mean.clamp(math.log(cfg.drift_min), math.log(cfg.drift_max))
            parent_hits = int((bounded_upper != upper_mean).sum())
            self.parent_mean, self.parent_cov = bounded_upper, upper_cov
        else:
            j3 = 0.0
        lower = torch.full_like(updated_mean, math.log(cfg.q_min))
        upper = torch.full_like(updated_mean, math.log(cfg.q_max))
        lower[-1], upper[-1] = math.log(cfg.r_min), math.log(cfg.r_max)
        bounded = updated_mean.maximum(lower).minimum(upper)
        self.mean, self.cov = bounded, updated_cov
        self.last_diagnostics = {
            "hyper_bound_hits": int((bounded != updated_mean).sum()),
            "parent_bound_hits": parent_hits,
            "hyper_jitter": max(j1, j2, j3),
            "hyper_score_norm": float(score.norm()),
            "hyper_drift": drift.cpu().tolist(),
            **information_diagnostics,
        }
        if not all(torch.isfinite(t).all() for t in (self.mean, self.cov, self.parent_mean, self.parent_cov)):
            raise FloatingPointError("Nonfinite uncertainty state")

    def state_dict(self) -> dict[str, Any]:
        """Serialize all persistent inference state for audit or later continuation."""
        return {
            "method": self.method, "config": asdict(self.config),
            "mean": self.mean.cpu(), "cov": self.cov.cpu(),
            "parent_mean": self.parent_mean.cpu(), "parent_cov": self.parent_cov.cpu(),
        }


class DiagonalEKF:
    """One simultaneous minibatch update with a diagonal parameter posterior."""

    def __init__(self, groups: list[int], method: str, config: FilterConfig, device: torch.device):
        """Groups describe consecutive, nonempty blocks in the parameter vector."""
        if not groups or min(groups) < 1:
            raise ValueError("Parameter group sizes must be positive")
        self.groups, self.config = groups, config
        self.p = torch.full((sum(groups),), config.initial_p, device=device, dtype=torch.float32)
        self.controller = UncertaintyController(method, len(groups), config, device)

    @torch.no_grad()
    def step(self, residual: Tensor, jacobian: Tensor) -> tuple[Tensor, dict[str, Any]]:
        """Return a parameter displacement; freeze linearization for the whole batch."""
        if residual.ndim != 1 or jacobian.shape != (residual.numel(), self.p.numel()):
            raise ValueError("Expected residual[M] and Jacobian[M, P]")
        if residual.dtype != torch.float32 or jacobian.dtype != torch.float32:
            raise TypeError("Filter inputs must be FP32")
        if not torch.isfinite(residual).all() or not torch.isfinite(jacobian).all():
            raise FloatingPointError("Nonfinite filter input")
        q, r = self.controller.prediction()
        q_vector = torch.cat([q[i].expand(size) for i, size in enumerate(self.groups)])
        prior = self.p + q_vector
        work_residual, work_jacobian = residual, jacobian
        initial_jitter, cancellation_candidates, factorization_failed = 0.0, 0, False
        noise_resolution_risk = False
        try:
            displacement, post, innovation_cov, factor, jitter = condition_diagonal(residual, jacobian, prior, r)
            initial_jitter = jitter
            cancellation_candidates = int(((post / prior < 1e-3) | (post <= 0)).sum())
            noise_resolution_risk = float(r) <= 8 * torch.finfo(residual.dtype).eps * float(innovation_cov.diagonal().max())
            precision_rescue = cancellation_candidates > 0 or jitter > 0 or noise_resolution_risk
        except FloatingPointError:
            precision_rescue, factorization_failed = True, True
        if precision_rescue:
            rescue_device = torch.device("cpu") if residual.device.type == "mps" else residual.device
            work_residual = residual.to(device=rescue_device, dtype=torch.float64)
            work_jacobian = jacobian.to(device=rescue_device, dtype=torch.float64)
            prior = self.p.to(work_residual) + q_vector.to(work_residual)
            displacement, post, innovation_cov, factor, jitter = condition_diagonal(
                work_residual, work_jacobian, prior, r.to(work_residual)
            )
        bounded = post.clamp(self.config.p_floor, self.config.p_ceiling)
        identity = torch.eye(work_residual.numel(), device=work_residual.device, dtype=work_residual.dtype)
        work_q, work_r = q.to(work_residual), r.to(work_residual)
        derivatives = []
        offset = 0
        for index, size in enumerate(self.groups):
            block = work_jacobian[:, offset:offset + size]
            derivatives.append(work_q[index] * (block @ block.mT))
            offset += size
        derivatives.append(work_r * identity)
        self.controller.observe(work_residual, innovation_cov, torch.stack(derivatives))
        self.p = bounded.to(self.p)
        displacement = displacement.to(residual)
        nll = 0.5 * (
            work_residual @ torch.cholesky_solve(work_residual[:, None], factor).squeeze(1)
            + 2 * factor.diagonal().log().sum()
            + residual.numel() * math.log(2 * math.pi)
        ) / residual.numel()
        diagnostics = {
            "q_used": q.cpu().tolist(), "r_used": float(r),
            "innovation_nll_per_component": float(nll),
            "update_norm": float(displacement.norm()),
            "p_mean": float(self.p.mean()), "p_bound_hits": int((post != bounded).sum()),
            "innovation_jitter": jitter,
            "innovation_initial_jitter": initial_jitter,
            "covariance_precision_rescue": int(precision_rescue),
            "covariance_cancellation_candidates": cancellation_candidates,
            "innovation_initial_factorization_failed": int(factorization_failed),
            "innovation_noise_resolution_risk": int(noise_resolution_risk),
            **self.controller.last_diagnostics,
        }
        if not torch.isfinite(displacement).all() or not torch.isfinite(self.p).all():
            raise FloatingPointError("Nonfinite EKF posterior")
        return displacement, diagnostics

    def state_dict(self) -> dict[str, Any]:
        """Return detached CPU inference state."""
        return {"groups": self.groups, "p": self.p.cpu(), "controller": self.controller.state_dict()}
