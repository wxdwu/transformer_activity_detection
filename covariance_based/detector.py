from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CovarianceMLEResult:
    activities: np.ndarray
    objective_history: list[float]
    sweeps: int
    converged: bool
    max_coordinate_step: float


def covariance_mle_activity(
    pilot_matrix: np.ndarray,
    sample_covariance: np.ndarray,
    noise_variance: float,
    *,
    max_sweeps: int = 100,
    tolerance: float = 1e-4,
    random_order: bool = True,
    rng: np.random.Generator | None = None,
) -> CovarianceMLEResult:
    """Minimize logdet(Sigma) + trace(inv(Sigma) @ C) for activities in [0, 1].

    Sigma = B @ diag(activities) @ B.conj().T + noise_variance * I.
    B is the original scaled complex pilot matrix [L, N], not normalized x_b.
    C is Y @ Y.conj().T / M. No labels, activity prior, or geometry are used.

    This is the known-large-scale-fading, box-constrained covariance baseline
    described in Li et al., IEEE TWC, 2023, DOI: 10.1109/TWC.2022.3218579.
    """
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
    if not isinstance(max_sweeps, int) or max_sweeps < 1:
        raise ValueError("max_sweeps must be a positive integer.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")

    pilot_energy = np.sum(np.abs(b) ** 2, axis=0)
    scale = max(float(pilot_energy.mean()), float(np.trace(c).real / pilot_len), noise_variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Input power must be finite and positive.")
    # Scale every physical power together; independent normalization changes the model.
    b = b / np.sqrt(scale)
    c = c / scale
    if not np.allclose(c, c.conj().T, rtol=1e-7, atol=1e-10):
        raise ValueError("sample_covariance must be Hermitian.")
    c = (c + c.conj().T) * 0.5
    if np.linalg.eigvalsh(c).min() < -1e-9 * max(1.0, float(np.linalg.norm(c, ord=2))):
        raise ValueError("sample_covariance must be positive semidefinite.")
    noise = noise_variance / scale
    eye = np.eye(pilot_len, dtype=np.complex128)
    activities = np.zeros(num_devices, dtype=np.float64)
    valid = np.flatnonzero(pilot_energy > 0)
    rng = np.random.default_rng() if rng is None else rng
    objective_offset = pilot_len * np.log(scale)

    def rebuild() -> tuple[np.ndarray, float]:
        sigma = (b * activities) @ b.conj().T + noise * eye
        sigma = (sigma + sigma.conj().T) * 0.5
        chol = np.linalg.cholesky(sigma)
        inverse = np.linalg.solve(chol.conj().T, np.linalg.solve(chol, eye))
        inverse = (inverse + inverse.conj().T) * 0.5
        objective = 2.0 * np.log(chol.diagonal().real).sum() + np.trace(inverse @ c).real
        if not np.isfinite(objective) or not np.isfinite(inverse).all():
            raise FloatingPointError("Nonfinite covariance likelihood or inverse.")
        return inverse, float(objective + objective_offset)

    def stationary_step(inverse: np.ndarray) -> float:
        if not len(valid):
            return 0.0
        pilots = b[:, valid]
        vectors = inverse @ pilots
        q = np.sum(pilots.conj() * vectors, axis=0).real
        r = np.sum(vectors.conj() * (c @ vectors), axis=0).real
        if np.any(q <= 0) or not np.isfinite(q).all() or not np.isfinite(r).all():
            raise FloatingPointError("Invalid coordinate curvature.")
        delta = (np.maximum(r, 0.0) / q - 1.0) / q
        return float(np.max(np.abs(np.clip(activities[valid] + delta, 0.0, 1.0) - activities[valid])))

    inverse, objective = rebuild()
    history = [objective]
    max_step = stationary_step(inverse)
    if max_step <= tolerance:
        return CovarianceMLEResult(activities, history, 0, True, max_step)

    for sweep in range(1, max_sweeps + 1):
        order = rng.permutation(valid) if random_order else valid
        for index in order:
            pilot = b[:, index]
            vector = inverse @ pilot
            q = float(np.vdot(pilot, vector).real)
            r = float(np.vdot(vector, c @ vector).real)
            if q <= 0 or not np.isfinite(q) or not np.isfinite(r):
                raise FloatingPointError("Invalid coordinate curvature.")
            delta = (max(r, 0.0) / q - 1.0) / q
            updated = float(np.clip(activities[index] + delta, 0.0, 1.0))
            delta = updated - activities[index]
            if delta == 0.0:
                continue
            activities[index] = updated
            denominator = 1.0 + delta * q
            if denominator <= np.sqrt(np.finfo(np.float64).eps):
                inverse, _ = rebuild()
            else:
                inverse -= (delta / denominator) * np.outer(vector, vector.conj())

        # Refresh the inverse each sweep to control accumulated rank-one-update error.
        inverse, objective = rebuild()
        history.append(objective)
        max_step = stationary_step(inverse)
        if max_step <= tolerance:
            return CovarianceMLEResult(activities, history, sweep, True, max_step)

    return CovarianceMLEResult(activities, history, max_sweeps, False, max_step)
