# Mathematical contract and provenance

## What is implemented

The proposed controller uses a hierarchy of Gaussian beliefs over log variances.
It extends a diagonal EKF for small CNN adapter parameters with a local
innovation-likelihood Fisher approximation. It is **not** a literal port of the
generalized-HGF node update equations, PyHGF, or LoKO. The CNN still supplies
ordinary automatic derivatives of its outputs.

The generalized HGF paper describes the coupling concept as:

> "parent nodes either predict the state of child nodes or their rate of change."

[Weber et al., Abstract, generalized HGF](https://arxiv.org/html/2305.10937v3)

This implementation transfers that hierarchy to the variance of log-Q/log-R
states. The update below is our explicit approximation for this proposed
experiment. It has not been shown equivalent to the original HGF algorithm.

LoKO states:

> "with no process noise"
>
> "we adopt a diagonal approximation of the covariance matrix"

[Abdi et al., Section 4](https://arxiv.org/html/2410.11551v2)

Those are the relevant starting assumptions. The `zero_q` arm is a controlled
LoKO-inspired comparison, not an exact published reproduction: our adapters
are residual modules, R is scalar-isotropic, and its residual EMA is used from
the next minibatch. No claim is made to reproduce LoKO's reported results.

## Observation and parameter model

One time step is one complete minibatch; there is no update between its images.
The state theta contains adapter weights, grouped by adapter block g:

```
theta_t = theta_(t-1) + process_noise
Q_t[j,j] = exp(z_q[g]) for parameter j in group g
y_t = softmax(CNN_theta(x_t)) + observation_noise
R_t = exp(z_r) I
```

The one-hot y and Gaussian likelihood form a surrogate classification objective.
R is effective output residual variance, not a direct estimate of camera noise.
Q measures drift in the chosen parameter coordinates, not physical scene motion.
The low-rank factors also have a change-of-basis ambiguity; this pilot fixes
their parameterization and initialization across methods, but does not establish
gauge invariance. A fixed-projection adapter is a possible later control.

Let p be the diagonal parameter posterior variance, H the Jacobian of flattened
probabilities, and e the one-hot residual, all at the same pre-update theta:

```
p_minus = p_previous + diag(Q)
S = H diag(p_minus) H^T + R
K = diag(p_minus) H^T S^-1
theta_plus = theta_minus + K e
p_plus = diag(diag(p_minus) - K H diag(p_minus))
```

The full posterior is projected onto its diagonal after each minibatch. No
parameter-by-parameter covariance matrix is allocated. Outputs across the
minibatch remain jointly conditioned through S. Process variance is per update;
changing batch size changes the model's time scale and requires retuning.

## Lower uncertainty state

z contains one log-Q per adapter and one log-R. Its covariance C is a full small
matrix, so evidence that could be explained by either Q or R is not incorrectly
treated as independent. For the nonhierarchical `adaptive_qr` arm, the diffusion
of z is a fixed diagonal matrix. For `hierarchical_qr`, each diffusion is exp(u_i).
Both arms otherwise use identical initial states and calculations.

```
C_minus = C_previous + diag(exp(u))
D_i = dS/dz_i
g_i = 0.5 [e^T S^-1 D_i S^-1 e - tr(S^-1 D_i)]
F_ij = 0.5 tr(S^-1 D_i S^-1 D_j)
C_plus = (C_minus^-1 + F)^-1
z_plus = z_minus + C_plus g
```

For a Q group, D_i = Q_i H_i H_i^T. For R, D_i = R I. This is one Fisher-scoring
Gaussian update using the predictive likelihood, not exact joint integration
over theta and z. Correlated/proportional D matrices can make Q/R unidentifiable;
the code logs Fisher diagnostics rather than claiming automatic separation.

## Upper uncertainty state

The parent u has a Gaussian random-walk prior with a fixed small covariance
increment. Treat the lower local quadratic likelihood as a Gaussian
pseudo-observation and integrate z out. Define:

```
B = C_minus^-1
A = C_plus
d = z_plus - z_minus
J = B - B A B
v = B d
G_i = exp(u_i) e_i e_i^T
g_upper_i = 0.5 [v^T G_i v - tr(J G_i)]
F_upper_ij = 0.5 tr(J G_i J G_j)
```

The parent mean/covariance use the same Gaussian score update. With no lower
information, A=C_minus and d=0, hence J=0: the parent gains no spurious evidence.
Using B instead of J in its Fisher matrix would incorrectly shrink uncertainty
even when the lower observation was uninformative. Independent finite-difference
and numerical-integration tests check these expressions.

## Causal ordering

1. Predict Q/R from previous beliefs.
2. Predict and score the current images before their labels update anything.
3. Compute the parameter update with those predicted Q/R values.
4. Use the same pre-update innovation to update z and u for future steps.
5. Commit the parameter displacement exactly once.

Current labels update every posterior in the hierarchy, but the newly inferred
Q/R never get reused to recompute the current parameter gain. New Q/R begin to
affect parameters on the next step. Domain-boundary metadata never enters any
controller.

## Numerical and inferential limitations

- Parameter posterior correlations are discarded after conditioning.
- The innovation model is locally linear in parameters.
- Fisher scoring replaces observed, possibly indefinite curvature.
- Log variance bounds are a projection; covariance is not the covariance of a
  properly truncated distribution. Bound hits are logged and must be inspected.
- Cholesky jitter is bounded and reported, rather than silently escalating.
- Highly informative observations can cause FP32 subtraction cancellation.
  A detected fragile update is recomputed with temporary FP64 conditioning and
  reported as a precision rescue. Persistent state stays FP32.
- Parent and child estimates are not independent scientific measurements of
  ground-truth visual volatility and noise.
- The default hyperparameters are not a completed tuning study.

The test suite checks algebra, derivatives, causal ordering, identity
initialization, frozen BatchNorm/backbone, disjoint data, and failure reporting.
Those checks cannot establish empirical superiority, novelty, or venue fitness.
