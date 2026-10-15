from __future__ import annotations

from dataclasses import dataclass

import numpy as np


NNLS_MAX_ITER = 1000
NNLS_TOLERANCE = 1e-10


@dataclass
class CovarianceNNLSResult:
    activities: np.ndarray
    iterations: int
    converged: bool
    optimality: float
    projected_gradient_norm: float
    scaled_residual_squared: float
    power_scale: float
    status: int


def covariance_nnls_activity(
    pilot_matrix: np.ndarray,
    sample_covariance: np.ndarray,
    noise_variance: float,
    *,
    max_iter: int = NNLS_MAX_ITER,
    tolerance: float = NNLS_TOLERANCE,
) -> CovarianceNNLSResult:
    """Fit C = B diag(a) B^H + noise*I with 0 <= a <= 1 using SciPy BVLS.

    This is box-constrained NNLS for known scaled pilots, not unbounded NNLS.
    The squared complex Frobenius residual is realified without dropping
    imaginary or diagonal information. No activity prior, labels or G is used.
    Residual and optimality diagnostics use a common physical-power scaling;
    the original squared residual equals scaled_residual_squared*power_scale**2.
    """
    from scipy.optimize import lsq_linear

    b = np.asarray(pilot_matrix, dtype=np.complex128)
    c = np.asarray(sample_covariance, dtype=np.complex128)
    noise_variance = float(noise_variance)
    if b.ndim != 2 or min(b.shape) == 0:
        raise ValueError("pilot_matrix must have nonempty shape [L, N].")
    pilot_len, num_devices = b.shape
    if c.shape != (pilot_len, pilot_len):
        raise ValueError("sample_covariance must have shape [L, L].")
    if not np.isfinite(b).all() or not np.isfinite(c).all():
        raise ValueError("Pilots and covariance must contain only finite values.")
    if not np.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError("noise_variance must be finite and positive.")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")

    scale = max(float(np.sum(np.abs(b) ** 2, axis=0).mean()),
                float(np.trace(c).real / pilot_len), noise_variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Input power must be finite and positive.")
    b = b / np.sqrt(scale)
    c = c / scale
    if not np.allclose(c, c.conj().T, rtol=1e-7, atol=1e-10):
        raise ValueError("sample_covariance must be Hermitian.")
    c = (c + c.conj().T) * 0.5
    if np.linalg.eigvalsh(c).min() < -1e-9 * max(1.0, float(np.linalg.norm(c, ord=2))):
        raise ValueError("sample_covariance must be positive semidefinite.")

    atoms = (b[:, None, :] * b.conj()[None, :, :]).reshape(pilot_len**2, num_devices)
    target = (c - (noise_variance / scale) * np.eye(pilot_len)).reshape(-1)
    design = np.vstack((atoms.real, atoms.imag))
    observed = np.concatenate((target.real, target.imag))
    result = lsq_linear(design, observed, bounds=(0.0, 1.0), method="bvls",
                        lsq_solver="exact", tol=tolerance, max_iter=max_iter)
    if not np.isfinite(result.x).all() or not np.isfinite(result.cost) or not np.isfinite(result.optimality):
        raise FloatingPointError("Nonfinite bounded NNLS solution or diagnostics.")
    if np.any(result.x < -1e-10) or np.any(result.x > 1.0 + 1e-10):
        raise FloatingPointError("Bounded NNLS returned an infeasible solution.")
    # BVLS solves the bounded problem; only snap floating-point boundary drift.
    activities = np.clip(result.x, 0.0, 1.0)
    residual = design @ activities - observed
    gradient = design.T @ residual
    projected_gradient_norm = float(np.max(np.abs(activities - np.clip(activities - gradient, 0.0, 1.0))))
    return CovarianceNNLSResult(
        activities=activities,
        iterations=int(result.nit),
        converged=bool(result.success and projected_gradient_norm <= tolerance),
        optimality=float(result.optimality),
        projected_gradient_norm=projected_gradient_norm,
        scaled_residual_squared=float(np.dot(residual, residual)),
        power_scale=scale,
        status=int(result.status),
    )
