from __future__ import annotations

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from covariance_based.nnls import covariance_nnls_activity


class CovarianceNNLSTests(unittest.TestCase):
    def test_scalar_closed_form_and_bounds(self) -> None:
        for power, expected in ((0.1, 0.0), (1.8, 0.4), (8.0, 1.0)):
            result = covariance_nnls_activity(np.array([[2.0j]]), np.array([[power]]), 0.2)
            np.testing.assert_allclose(result.activities, [expected], atol=1e-12)
            self.assertTrue(result.converged)

    def test_exact_population_covariance(self) -> None:
        b = np.diag([1.0, 2.0j, 0.5, 1.2 - 0.3j])
        activity = np.array([0.0, 1.0, 0.4, 0.0])
        c = (b * activity) @ b.conj().T + 0.1 * np.eye(4)
        result = covariance_nnls_activity(b, c, 0.1)
        np.testing.assert_allclose(result.activities, activity, atol=1e-12)
        self.assertTrue(result.converged)
        self.assertLess(result.scaled_residual_squared, 1e-24)

    def test_feasible_unconstrained_solution_requires_no_iterations(self) -> None:
        result = covariance_nnls_activity(np.eye(3), np.diag([0.3, 0.5, 0.8]), 0.1)
        np.testing.assert_allclose(result.activities, [0.2, 0.4, 0.7], atol=1e-12)
        self.assertEqual(result.iterations, 0)
        self.assertTrue(result.converged)

    def test_bounded_fit_reoptimizes_instead_of_clipping(self) -> None:
        b = np.array([[1.0, 1.0], [0.0, 1.0]])
        c = (b * [1.8, 0.2]) @ b.T + 0.1 * np.eye(2)
        result = covariance_nnls_activity(b, c, 0.1)
        np.testing.assert_allclose(result.activities, [1.0, 0.4], atol=1e-10)
        self.assertFalse(np.allclose(result.activities, [1.0, 0.2]))

    def test_complex_nonorthogonal_fit_matches_independent_active_set_enumeration(self) -> None:
        b = np.array([[1.0, 1.0j, 0.3], [0.2j, 0.7, 1.0]])
        y = np.array([[1.3, 0.2j, 0.4], [0.1j, 0.7, 0.6j]])
        c = y @ y.conj().T + 0.1 * np.eye(2)
        target = c - 0.1 * np.eye(2)
        gram = np.abs(b.conj().T @ b)**2
        linear = np.diag(b.conj().T @ target @ b).real
        candidates = []
        for state in itertools.product((-1, 0, 1), repeat=3):
            free = np.flatnonzero(np.array(state) == 0)
            fixed = np.flatnonzero(np.array(state) != 0)
            activity = (np.array(state) == 1).astype(float)
            if free.size:
                rhs = linear[free] - gram[np.ix_(free, fixed)] @ activity[fixed]
                activity[free] = np.linalg.solve(gram[np.ix_(free, free)], rhs)
            if np.all((activity >= 0) & (activity <= 1)):
                error = np.linalg.norm((b * activity) @ b.conj().T - target, "fro")**2
                candidates.append((error, activity))
        best_error, best_activity = min(candidates, key=lambda row: row[0])
        result = covariance_nnls_activity(b, c, 0.1, tolerance=1e-10)
        np.testing.assert_allclose(result.activities, best_activity, atol=1e-8)
        self.assertAlmostEqual(result.scaled_residual_squared * result.power_scale**2, best_error, places=10)
        self.assertTrue(result.converged)
        self.assertLess(result.optimality, 1e-8)

    def test_common_power_scaling_preserves_activity(self) -> None:
        b = np.array([[1.0, 0.3j, 0.2], [0.0, 1.0, 0.5j]])
        c = (b * [1.0, 0.0, 0.4]) @ b.conj().T + 0.1 * np.eye(2)
        reference = covariance_nnls_activity(b, c, 0.1)
        for scale in (1e-24, 1e12):
            result = covariance_nnls_activity(b * np.sqrt(scale), c * scale, 0.1 * scale)
            np.testing.assert_allclose(result.activities, reference.activities, atol=1e-9)
            self.assertAlmostEqual(result.scaled_residual_squared, reference.scaled_residual_squared, places=10)

    def test_empty_scene_and_zero_columns(self) -> None:
        b = np.column_stack((np.eye(3), np.zeros(3)))
        for pilots in (b, np.zeros_like(b)):
            result = covariance_nnls_activity(pilots, 0.2 * np.eye(3), 0.2)
            np.testing.assert_allclose(result.activities, np.zeros(4), atol=1e-12)
            self.assertTrue(result.converged)

    def test_rank_deficient_dictionary_is_valid(self) -> None:
        b = np.array([[1.0, 1.0], [0.5j, 0.5j]])
        c = (b * [0.2, 0.4]) @ b.conj().T + 0.1 * np.eye(2)
        result = covariance_nnls_activity(b, c, 0.1)
        np.testing.assert_allclose((b * result.activities) @ b.conj().T + 0.1 * np.eye(2), c, atol=1e-10)
        self.assertTrue(np.all((result.activities >= 0) & (result.activities <= 1)))

    def test_roundoff_is_snapped_but_real_bound_violations_are_rejected(self) -> None:
        for activity, invalid in (([-1e-15, 1 + 1e-15], False), ([-0.1, 1.1], True)):
            result = SimpleNamespace(x=np.array(activity), cost=0.0, optimality=0.0, nit=1, success=True, status=1)
            with patch("scipy.optimize.lsq_linear", return_value=result):
                if invalid:
                    with self.assertRaises(FloatingPointError):
                        covariance_nnls_activity(np.eye(2), np.diag([0.1, 1.1]), 0.1)
                else:
                    fitted = covariance_nnls_activity(np.eye(2), np.diag([0.1, 1.1]), 0.1)
                    np.testing.assert_array_equal(fitted.activities, [0, 1])

    def test_relative_cost_success_requires_projected_gradient_certificate(self) -> None:
        result = SimpleNamespace(x=np.array([0.2, 0.7]), cost=0.1, optimality=0.3, nit=1, success=True, status=2)
        with patch("scipy.optimize.lsq_linear", return_value=result):
            fitted = covariance_nnls_activity(np.eye(2), np.diag([0.1, 1.1]), 0.1)
        self.assertFalse(fitted.converged)
        self.assertGreater(fitted.projected_gradient_norm, 1e-10)

    def test_invalid_inputs(self) -> None:
        b = np.eye(2, dtype=complex)
        for noise in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                covariance_nnls_activity(b, np.eye(2), noise)
        for covariance in (np.eye(3), np.diag([1.0, -1.0]),
                           np.array([[1, 1j], [0, 1]]), np.full((2, 2), np.nan)):
            with self.assertRaises(ValueError):
                covariance_nnls_activity(b, covariance, 0.1)
        for options in ({"max_iter": 0}, {"max_iter": 1.5}, {"tolerance": 0}, {"tolerance": float("nan")}):
            with self.assertRaises(ValueError):
                covariance_nnls_activity(b, np.eye(2), 0.1, **options)
        with self.assertRaises(ValueError):
            covariance_nnls_activity(np.array([1.0]), np.eye(2), 0.1)


if __name__ == "__main__":
    unittest.main()
