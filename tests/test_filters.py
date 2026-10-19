"""Independent Gaussian conditioning and numerical likelihood derivative checks."""

from itertools import product
import math

import pytest
import torch

from hgfvision.filters import (
    DiagonalEKF,
    FilterConfig,
    UncertaintyController,
    cholesky,
    fisher_diagnostics,
    gaussian_score_update,
    parent_score_fisher,
    variance_score_fisher,
)


@pytest.fixture(scope="module", autouse=True)
def small_cpu_workload():
    """Restore global threading and random state after these small matrix tests."""
    threads = torch.get_num_threads()
    random_state = torch.get_rng_state()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(random_state)


def log_normal(residual, covariance):
    """Evaluate a normalized Gaussian density without production filter helpers."""
    sign, logdet = torch.linalg.slogdet(covariance)
    assert sign > 0
    return -0.5 * (
        residual @ torch.linalg.solve(covariance, residual)
        + logdet
        + residual.numel() * math.log(2 * math.pi)
    )


def central_gradient(function, point, step=1e-5):
    """Differentiate only scalar likelihood evaluations using central differences."""
    basis = torch.eye(point.numel(), dtype=point.dtype)
    return torch.stack([
        (function(point + step * direction) - function(point - step * direction)) / (2 * step)
        for direction in basis
    ])


def expected_numeric_score_outer(covariance_function, point):
    """Integrate numerical Gaussian scores with exact degree-five quadrature."""
    covariance = covariance_function(point)
    factor = torch.linalg.cholesky(covariance)
    nodes = [(0.0, 2 / 3), (-math.sqrt(3), 1 / 6), (math.sqrt(3), 1 / 6)]
    result = torch.zeros(point.numel(), point.numel(), dtype=point.dtype)
    for choices in product(nodes, repeat=covariance.shape[0]):
        node = torch.tensor([choice[0] for choice in choices], dtype=point.dtype)
        weight = math.prod(choice[1] for choice in choices)
        residual = factor @ node
        score = central_gradient(lambda value: log_normal(residual, covariance_function(value)), point)
        result += weight * torch.outer(score, score)
    return result


def test_variance_score_and_fisher_match_numerical_likelihood_derivatives():
    vectors = torch.tensor([[1.0, 0.4, -0.2], [0.3, -0.7, 0.8]], dtype=torch.float64)
    components = torch.stack([torch.outer(vector, vector) for vector in vectors] + [torch.eye(3, dtype=torch.float64)])
    base = torch.tensor([[0.6, 0.12, 0.0], [0.12, 0.5, -0.1], [0.0, -0.1, 0.4]], dtype=torch.float64)
    point = torch.tensor([-0.7, 0.2, -1.1], dtype=torch.float64)
    residual = torch.tensor([0.8, -1.2, 0.3], dtype=torch.float64)

    def covariance(value):
        return base + (value.exp()[:, None, None] * components).sum(0)

    derivatives = point.exp()[:, None, None] * components
    score, fisher, jitter = variance_score_fisher(residual, covariance(point), derivatives)
    expected_score = central_gradient(lambda value: log_normal(residual, covariance(value)), point)
    expected_fisher = expected_numeric_score_outer(covariance, point)
    torch.testing.assert_close(score, expected_score, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(fisher, expected_fisher, rtol=1e-7, atol=1e-9)
    assert jitter == 0


def test_parent_score_and_fisher_match_marginal_gaussian_finite_differences():
    base = torch.tensor([[0.7, 0.2, -0.1], [0.2, 0.5, 0.1], [-0.1, 0.1, 0.6]], dtype=torch.float64)
    design = torch.tensor([[0.9, -0.2, 0.4], [0.3, 1.1, -0.6]], dtype=torch.float64)
    noise = torch.tensor([[0.4, 0.1], [0.1, 0.6]], dtype=torch.float64)
    point = torch.tensor([-1.2, -0.8, -1.6], dtype=torch.float64)
    residual = torch.tensor([1.1, -0.7], dtype=torch.float64)

    def marginal_covariance(value):
        return design @ (base + torch.diag(value.exp())) @ design.mT + noise

    prior = base + torch.diag(point.exp())
    gain = torch.linalg.solve(marginal_covariance(point), design @ prior).mT
    posterior = prior - gain @ design @ prior
    displacement = gain @ residual
    score, fisher = parent_score_fisher(prior, posterior, displacement, point.exp())
    expected_score = central_gradient(lambda value: log_normal(residual, marginal_covariance(value)), point)
    expected_fisher = expected_numeric_score_outer(marginal_covariance, point)
    torch.testing.assert_close(score, expected_score, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(fisher, expected_fisher, rtol=1e-7, atol=1e-9)


def test_gaussian_score_update_matches_exact_linear_gaussian_conditioning():
    mean = torch.tensor([0.3, -0.4, 0.2], dtype=torch.float64)
    prior = torch.tensor([[0.8, 0.2, 0.1], [0.2, 0.6, -0.1], [0.1, -0.1, 0.9]], dtype=torch.float64)
    design = torch.tensor([[0.7, 1.0, -0.3], [0.2, -0.4, 0.8]], dtype=torch.float64)
    noise = torch.tensor([[0.4, -0.05], [-0.05, 0.3]], dtype=torch.float64)
    residual = torch.tensor([0.9, -0.6], dtype=torch.float64)
    innovation = design @ prior @ design.mT + noise
    gain = torch.linalg.solve(innovation, design @ prior).mT
    expected_mean = mean + gain @ residual
    remainder = torch.eye(3, dtype=torch.float64) - gain @ design
    expected_covariance = remainder @ prior @ remainder.mT + gain @ noise @ gain.mT
    score = design.mT @ torch.linalg.solve(noise, residual)
    fisher = design.mT @ torch.linalg.solve(noise, design)
    actual_mean, actual_covariance, jitter = gaussian_score_update(mean, prior, score, fisher)
    torch.testing.assert_close(actual_mean, expected_mean, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_covariance, expected_covariance, rtol=1e-12, atol=1e-12)
    assert jitter == 0


@pytest.mark.parametrize("dtype,atol", [(torch.float64, 1e-14), (torch.float32, 1e-6)])
def test_parent_has_no_evidence_without_lower_information(dtype, atol):
    prior = torch.tensor([[0.8, 0.2, -0.1], [0.2, 0.7, 0.1], [-0.1, 0.1, 0.9]], dtype=dtype)
    score, fisher = parent_score_fisher(prior, prior.clone(), torch.zeros(3, dtype=dtype), torch.full((3,), 0.02, dtype=dtype))
    torch.testing.assert_close(score, torch.zeros_like(score), rtol=0, atol=atol)
    torch.testing.assert_close(fisher, torch.zeros_like(fisher), rtol=0, atol=atol ** 2)


def test_hierarchical_controller_does_not_learn_from_flat_likelihood():
    config = FilterConfig()
    controller = UncertaintyController("hierarchical_qr", 2, config, torch.device("cpu"))
    mean_before = controller.mean.clone()
    parent_before = controller.parent_mean.clone()
    covariance_before = controller.cov.clone()
    parent_covariance_before = controller.parent_cov.clone()
    controller.observe(torch.tensor([0.8, -1.0]), torch.eye(2), torch.zeros(3, 2, 2))
    torch.testing.assert_close(controller.mean, mean_before, rtol=0, atol=1e-7)
    torch.testing.assert_close(controller.parent_mean, parent_before, rtol=0, atol=1e-7)
    torch.testing.assert_close(controller.cov, covariance_before + config.hyper_drift * torch.eye(3), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(controller.parent_cov, parent_covariance_before + config.parent_drift * torch.eye(3), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("method", ["zero_q", "fixed_q", "adaptive_qr", "hierarchical_qr"])
def test_scalar_ekf_matches_hand_conditioning(method):
    config = FilterConfig(initial_p=0.2, initial_q=0.03, initial_r=0.4)
    inference = DiagonalEKF([1], method, config, torch.device("cpu"))
    residual = torch.tensor([-0.7])
    jacobian = torch.tensor([[1.5]])
    prior = config.initial_p + (0 if method == "zero_q" else config.initial_q)
    variance = jacobian.item() ** 2 * prior + config.initial_r
    expected_delta = prior * jacobian.item() / variance * residual.item()
    expected_p = prior * config.initial_r / variance
    displacement, diagnostics = inference.step(residual, jacobian)
    assert displacement.item() == pytest.approx(expected_delta, rel=1e-6, abs=1e-7)
    assert inference.p.item() == pytest.approx(expected_p, rel=1e-6, abs=1e-7)
    assert diagnostics["r_used"] == pytest.approx(config.initial_r)


def test_diagonal_ekf_matches_dense_joseph_posterior_and_gaussian_nll():
    config = FilterConfig(initial_q=0.02, initial_r=0.3)
    inference = DiagonalEKF([2, 3], "adaptive_qr", config, torch.device("cpu"))
    inference.p = torch.tensor([0.1, 0.3, 0.2, 0.4, 0.15])
    inference.controller.mean[:2] = torch.tensor([0.02, 0.05]).log()
    jacobian = torch.tensor([[1.0, -0.2, 0.4, 0.1, 0.3], [0.3, 0.8, -0.5, 0.6, -0.2], [-0.2, 0.1, 0.4, 0.7, 0.9]])
    residual = torch.tensor([0.4, -0.8, 0.2])
    design = jacobian.double()
    prior = torch.diag(inference.p.double() + torch.tensor([0.02, 0.02, 0.05, 0.05, 0.05], dtype=torch.float64))
    noise = config.initial_r * torch.eye(3, dtype=torch.float64)
    innovation = design @ prior @ design.mT + noise
    gain = torch.linalg.solve(innovation, design @ prior).mT
    expected_delta = gain @ residual.double()
    remainder = torch.eye(5, dtype=torch.float64) - gain @ design
    expected_posterior = remainder @ prior @ remainder.mT + gain @ noise @ gain.mT
    displacement, diagnostics = inference.step(residual, jacobian)
    torch.testing.assert_close(displacement.double(), expected_delta, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(inference.p.double(), expected_posterior.diagonal(), rtol=1e-6, atol=1e-7)
    expected_nll = -log_normal(residual.double(), innovation).item() / residual.numel()
    assert diagnostics["innovation_nll_per_component"] == pytest.approx(expected_nll, rel=1e-6)
    assert (expected_posterior - torch.diag(expected_posterior.diagonal())).abs().max() > 0.01
    assert diagnostics["p_bound_hits"] == 0


@pytest.mark.parametrize("jacobian_scale", [1.0, 10.0, 100.0, 1000.0])
@pytest.mark.parametrize("method", ["fixed_q", "adaptive_qr", "hierarchical_qr"])
def test_precision_rescue_preserves_small_posterior_variance(jacobian_scale, method):
    config = FilterConfig(initial_r=1e-5, p_floor=1e-14)
    inference = DiagonalEKF([1], method, config, torch.device("cpu"))
    prior = float(inference.p[0]) + float(inference.controller.prediction()[0][0])
    noise = float(inference.controller.prediction()[1])
    expected_variance = prior * noise / (jacobian_scale ** 2 * prior + noise)
    displacement, diagnostics = inference.step(torch.tensor([0.5]), torch.tensor([[jacobian_scale]]))
    assert inference.p.item() == pytest.approx(expected_variance, rel=1e-5, abs=1e-15)
    assert diagnostics["covariance_precision_rescue"] == 1
    assert diagnostics["p_bound_hits"] == 0
    assert displacement.dtype == torch.float32
    for state in [inference.p, inference.controller.mean, inference.controller.cov, inference.controller.parent_cov]:
        assert state.dtype == torch.float32
        assert torch.isfinite(state).all()


def test_precision_rescue_uses_consistent_likelihood_when_noise_loses_fp32_resolution():
    config = FilterConfig(initial_r=1e-5, p_floor=1e-14)
    inference = DiagonalEKF([1], "adaptive_qr", config, torch.device("cpu"))
    residual = torch.tensor([0.2, -0.3])
    jacobian = torch.tensor([[100.0], [100.0]])
    prior = float(inference.p[0]) + float(inference.controller.prediction()[0][0])
    noise = float(inference.controller.prediction()[1])
    expected_variance = prior * noise / (20000 * prior + noise)
    covariance = prior * jacobian.double() @ jacobian.double().mT + noise * torch.eye(2, dtype=torch.float64)
    _, diagnostics = inference.step(residual, jacobian)
    assert diagnostics["covariance_precision_rescue"] == 1
    assert diagnostics["innovation_noise_resolution_risk"] == 1
    assert diagnostics["innovation_jitter"] == 0
    assert inference.p.item() == pytest.approx(expected_variance, rel=1e-5, abs=1e-15)
    expected_nll = -log_normal(residual.double(), covariance).item() / 2
    assert diagnostics["innovation_nll_per_component"] == pytest.approx(expected_nll, rel=1e-7)


def test_fisher_diagnostics_reveal_exact_covariance_confounding():
    _, fisher, _ = variance_score_fisher(torch.tensor([0.5]), torch.tensor([[0.7]]), torch.tensor([[[0.3]], [[0.4]]]))
    diagnostics = fisher_diagnostics(fisher)
    assert abs(diagnostics["fisher_min_eigenvalue"]) < 1e-7
    assert diagnostics["fisher_max_offdiagonal_correlation"] == pytest.approx(1.0, rel=1e-6)
    empty = fisher_diagnostics(torch.zeros(2, 2))
    assert empty["fisher_min_eigenvalue"] == 0
    assert empty["fisher_max_offdiagonal_correlation"] == 0


def test_adaptive_and_hierarchical_start_with_identical_lower_updates():
    config = FilterConfig(initial_q=0.01, initial_r=0.2)
    adaptive = DiagonalEKF([2, 1], "adaptive_qr", config, torch.device("cpu"))
    hierarchical = DiagonalEKF([2, 1], "hierarchical_qr", config, torch.device("cpu"))
    jacobian = torch.tensor([[0.3, 0.5, -0.2], [-0.4, 0.2, 0.8]])
    residual = torch.tensor([0.9, -0.3])
    delta_adaptive, diagnostics_adaptive = adaptive.step(residual, jacobian)
    delta_hierarchical, diagnostics_hierarchical = hierarchical.step(residual, jacobian)
    for first, second in [(delta_adaptive, delta_hierarchical), (adaptive.p, hierarchical.p), (adaptive.controller.mean, hierarchical.controller.mean), (adaptive.controller.cov, hierarchical.controller.cov)]:
        torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-7)
    assert diagnostics_adaptive["q_used"] == diagnostics_hierarchical["q_used"]
    assert diagnostics_adaptive["r_used"] == diagnostics_hierarchical["r_used"]


@pytest.mark.parametrize("method", ["fixed_q", "adaptive_qr", "hierarchical_qr"])
def test_present_residual_affects_variances_only_for_future_weight_updates(method):
    config = FilterConfig(initial_q=0.01, initial_r=0.2)
    first = DiagonalEKF([2, 1], method, config, torch.device("cpu"))
    second = DiagonalEKF([2, 1], method, config, torch.device("cpu"))
    jacobian = torch.tensor([[0.8, 0.2, -0.3], [0.1, -0.5, 0.7]])
    for residual in [torch.tensor([0.1, -0.2]), torch.tensor([-0.3, 0.4])]:
        first.step(residual, jacobian)
        second.step(residual, jacobian)
    residual = torch.tensor([0.6, -0.8])
    first_delta, first_diagnostics = first.step(residual, jacobian)
    second_delta, second_diagnostics = second.step(3 * residual, jacobian)
    torch.testing.assert_close(second_delta, 3 * first_delta, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(first.p, second.p, rtol=0, atol=0)
    assert first_diagnostics["q_used"] == second_diagnostics["q_used"]
    assert first_diagnostics["r_used"] == second_diagnostics["r_used"]
    assert not torch.allclose(first.controller.prediction()[1], second.controller.prediction()[1])


@pytest.mark.parametrize("method", ["zero_q", "fixed_q", "adaptive_qr", "hierarchical_qr"])
@pytest.mark.parametrize("jacobian_scale", [1e-3, 1.0, 100.0])
def test_repeated_updates_remain_finite_and_covariances_positive(method, jacobian_scale):
    generator = torch.Generator().manual_seed(902)
    config = FilterConfig()
    inference = DiagonalEKF([3, 5], method, config, torch.device("cpu"))
    for _ in range(20):
        jacobian = jacobian_scale * torch.randn(6, 8, generator=generator)
        residual = torch.randn(6, generator=generator).tanh()
        displacement, diagnostics = inference.step(residual, jacobian)
        assert torch.isfinite(displacement).all()
        assert torch.isfinite(inference.p).all()
        assert (inference.p >= config.p_floor).all()
        assert (inference.p <= config.p_ceiling).all()
        for covariance in [inference.controller.cov, inference.controller.parent_cov]:
            assert torch.isfinite(covariance).all()
            torch.testing.assert_close(covariance, covariance.mT, rtol=0, atol=1e-6)
            assert torch.linalg.eigvalsh(covariance.double()).min() > 0
        q, r = inference.controller.prediction()
        assert torch.isfinite(q).all() and torch.isfinite(r)
        assert float(r) >= config.r_min * (1 - 1e-6)
        assert float(r) <= config.r_max * (1 + 1e-6)
        assert math.isfinite(diagnostics["innovation_nll_per_component"])


def test_cholesky_reports_regularization_and_rejects_invalid_matrices():
    matrix = torch.ones(2, 2, dtype=torch.float64)
    factor, jitter = cholesky(matrix)
    assert jitter > 0
    torch.testing.assert_close(factor @ factor.mT, matrix + jitter * torch.eye(2, dtype=torch.float64))
    with pytest.raises(FloatingPointError, match="positive definite"):
        cholesky(torch.tensor([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        cholesky(torch.tensor([[float("nan")]]))


@pytest.mark.parametrize("residual,jacobian,error", [
    (torch.zeros(2, 1), torch.zeros(2, 3), ValueError),
    (torch.zeros(2), torch.zeros(2, 4), ValueError),
    (torch.zeros(2, dtype=torch.float64), torch.zeros(2, 3), TypeError),
    (torch.zeros(2), torch.zeros(2, 3, dtype=torch.float64), TypeError),
    (torch.tensor([0.0, float("nan")]), torch.zeros(2, 3), FloatingPointError),
    (torch.zeros(2), torch.full((2, 3), float("inf")), FloatingPointError),
])
def test_ekf_rejects_invalid_input_without_changing_state(residual, jacobian, error):
    inference = DiagonalEKF([2, 1], "hierarchical_qr", FilterConfig(), torch.device("cpu"))
    p_before = inference.p.clone()
    mean_before = inference.controller.mean.clone()
    with pytest.raises(error):
        inference.step(residual, jacobian)
    torch.testing.assert_close(inference.p, p_before, rtol=0, atol=0)
    torch.testing.assert_close(inference.controller.mean, mean_before, rtol=0, atol=0)
