from __future__ import annotations

import unittest
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from covariance_based.detector import covariance_mle_activity


class CovarianceMLETests(unittest.TestCase):
    def test_scalar_closed_form_and_box_bounds(self) -> None:
        b = np.array([[2.0j]])
        for power, expected in ((0.2, 0.0), (1.8, 0.4), (8.0, 1.0)):
            result = covariance_mle_activity(b, np.array([[power]]), 0.2, random_order=False)
            np.testing.assert_allclose(result.activities, [expected], atol=1e-12)
            self.assertTrue(result.converged)

    def test_exact_population_covariance(self) -> None:
        b = np.diag(np.array([1.0, 2.0j, 0.5, 1.2 - 0.3j]))
        activity = np.array([0.0, 1.0, 0.4, 0.0])
        c = (b * activity) @ b.conj().T + 0.1 * np.eye(4)
        result = covariance_mle_activity(b, c, 0.1, rng=np.random.default_rng(1))
        np.testing.assert_allclose(result.activities, activity, atol=1e-12)
        self.assertLessEqual(result.sweeps, 1)

    def test_likelihood_decreases_for_nonorthogonal_pilots(self) -> None:
        rng = np.random.default_rng(4)
        b = (rng.normal(size=(5, 12)) + 1j * rng.normal(size=(5, 12))) / np.sqrt(10)
        h = (rng.normal(size=(12, 48)) + 1j * rng.normal(size=(12, 48))) / np.sqrt(2)
        active = np.zeros(12)
        active[[1, 6, 9]] = 1
        w = (rng.normal(size=(5, 48)) + 1j * rng.normal(size=(5, 48))) * np.sqrt(0.03 / 2)
        y = (b * active) @ h + w
        c = y @ y.conj().T / 48
        result = covariance_mle_activity(b, c, 0.03, max_sweeps=200, rng=np.random.default_rng(9))
        self.assertTrue(np.isfinite(result.activities).all())
        self.assertTrue(np.all((result.activities >= 0) & (result.activities <= 1)))
        self.assertTrue(np.all(np.diff(result.objective_history) <= 1e-8))
        sigma = (b * result.activities) @ b.conj().T + 0.03 * np.eye(5)
        objective = np.linalg.slogdet(sigma)[1] + np.trace(np.linalg.solve(sigma, c)).real
        self.assertAlmostEqual(result.objective_history[-1], objective, places=9)

    def test_common_physical_rescaling_preserves_activities(self) -> None:
        b = np.array([[1.0, 0.3j, 0.2], [0.0, 1.0, 0.5j]])
        c = (b * [1.0, 0.0, 0.4]) @ b.conj().T + 0.1 * np.eye(2)
        reference = covariance_mle_activity(b, c, 0.1, random_order=False)
        for power_scale in (1e-24, 1e12):
            result = covariance_mle_activity(b * np.sqrt(power_scale), c * power_scale,
                                             0.1 * power_scale, random_order=False)
            np.testing.assert_allclose(result.activities, reference.activities, atol=1e-10)

    def test_all_inactive_and_zero_pilot_column(self) -> None:
        b = np.column_stack((np.eye(3), np.zeros(3))).astype(complex)
        result = covariance_mle_activity(b, 0.2 * np.eye(3), 0.2)
        np.testing.assert_allclose(result.activities, np.zeros(4), atol=1e-12)
        self.assertTrue(result.converged)

    def test_seed_reproducibility(self) -> None:
        b = np.array([[1.0, 0.3j, 0.2], [0.0, 1.0, 0.5j]])
        c = (b * [1.0, 0.0, 0.4]) @ b.conj().T + 0.1 * np.eye(2)
        a = covariance_mle_activity(b, c, 0.1, rng=np.random.default_rng(11))
        other = covariance_mle_activity(b, c, 0.1, rng=np.random.default_rng(11))
        np.testing.assert_array_equal(a.activities, other.activities)
        self.assertEqual(a.objective_history, other.objective_history)

    def test_invalid_inputs(self) -> None:
        b = np.eye(2, dtype=complex)
        for noise in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                covariance_mle_activity(b, np.eye(2), noise)
        with self.assertRaises(ValueError):
            covariance_mle_activity(b, np.eye(3), 0.1)
        with self.assertRaises(ValueError):
            covariance_mle_activity(b, np.diag([1.0, -1.0]), 0.1)
        with self.assertRaises(ValueError):
            covariance_mle_activity(b, np.array([[1, 1j], [0, 1]]), 0.1)


@unittest.skipUnless(find_spec("matplotlib") and find_spec("torch"), "Plot tests require matplotlib and torch.")
class CovariancePlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import matplotlib
        matplotlib.use("Agg")
        from covariance_based.run_monte_carlo import plot_results
        cls.plot_results = staticmethod(plot_results)

    def test_pm_pf_axes_match_reference_and_preserve_data(self) -> None:
        summaries = [{"sweep_value": ""}]
        curves = [[{"PM": 0.0, "PF": 1.0}, {"PM": 0.2, "PF": 0.03},
                   {"PM": 1.0, "PF": 0.0}]]
        original = deepcopy(curves)
        with TemporaryDirectory() as directory, patch("matplotlib.figure.Figure.savefig", autospec=True) as save:
            self.plot_results(Path(directory), summaries, curves, "none")
        ax = save.call_args_list[0].args[0].axes[0]
        self.assertEqual((ax.get_xscale(), ax.get_yscale()), ("log", "log"))
        self.assertIn("P_M", ax.get_xlabel())
        self.assertIn("P_F", ax.get_ylabel())
        np.testing.assert_allclose(ax.lines[0].get_xdata(), [np.nan, 0.2, 1.0], equal_nan=True)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [1.0, 0.03, np.nan], equal_nan=True)
        self.assertEqual(curves, original)
        self.assertEqual(save.call_count, 2)

    def test_parameter_sweeps_keep_linear_x_and_log_probability_y(self) -> None:
        for sweep in ("L", "N", "SNR"):
            with self.subTest(sweep=sweep):
                summaries = [{"sweep_value": 4, "PM": 1e-6, "PF": 0.0},
                             {"sweep_value": 16, "PM": 0.03, "PF": 0.0}]
                curves = [[{"PM": 0.1, "PF": 0.2}]] * 2
                with TemporaryDirectory() as directory, patch("matplotlib.figure.Figure.savefig", autospec=True) as save:
                    self.plot_results(Path(directory), summaries, curves, sweep)
                axes = save.call_args_list[2].args[0].axes
                for ax in axes:
                    self.assertEqual((ax.get_xscale(), ax.get_yscale()), ("linear", "log"))
                    self.assertIn("10^", ax.yaxis.get_major_formatter()(1e-3))
                    self.assertEqual(ax.get_xlim(), (4.0, 16.0))
                    np.testing.assert_array_equal(ax.get_xticks(), [4, 16])
                self.assertLessEqual(axes[0].get_ylim()[0], 1e-6)
                self.assertTrue(np.isnan(axes[1].lines[0].get_ydata()[1]))
                self.assertTrue(np.isnan(axes[1].lines[0].get_ydata()).all())
                self.assertEqual(save.call_count, 4)

    def test_zero_and_undefined_probabilities_do_not_create_fake_positive_values(self) -> None:
        summaries = [{"sweep_value": ""}]
        curves = [[{"PM": 0.0, "PF": 0.0}, {"PM": float("nan"), "PF": float("nan")}]]
        with TemporaryDirectory() as directory, patch("matplotlib.figure.Figure.savefig", autospec=True) as save:
            self.plot_results(Path(directory), summaries, curves, "none")
        ax = save.call_args_list[0].args[0].axes[0]
        self.assertTrue(np.isnan(ax.lines[0].get_xdata()).all())
        self.assertTrue(np.isnan(ax.lines[0].get_ydata()).all())
        self.assertGreater(ax.get_xlim()[0], 0.0)
        self.assertGreater(ax.get_ylim()[0], 0.0)


if __name__ == "__main__":
    unittest.main()
