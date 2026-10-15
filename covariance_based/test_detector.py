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


@unittest.skipUnless(find_spec("torch"), "Configuration tests require torch.")
class CovarianceExperimentConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import covariance_based.run_monte_carlo as runner
        cls.runner = runner

    def test_temporary_settings_and_restoration(self) -> None:
        from dataclasses import asdict
        runner = self.runner
        inherited = runner.build_system_config(runner.build_args())
        with patch.object(runner, "USE_PAPER_EVENT_TEST_SETTINGS", True), patch.object(runner, "PILOT_LEN_OVERRIDE", None):
            cfg = runner.experiment_system(runner.parse_args([]))
        self.assertEqual(asdict(cfg), {**asdict(inherited), **runner.PAPER_EVENT_TEST_SETTINGS})
        self.assertEqual(cfg.activity_mode, "event")
        self.assertEqual(cfg.event_sigma_m, inherited.event_sigma_m)
        with patch.object(runner, "USE_PAPER_EVENT_TEST_SETTINGS", False), patch.object(runner, "PILOT_LEN_OVERRIDE", None):
            restored = runner.experiment_system(runner.parse_args([]))
        self.assertEqual(asdict(restored), asdict(inherited))

    def test_default_monte_carlo_inherits_transformer_scene(self) -> None:
        from dataclasses import asdict
        runner = self.runner
        args = runner.parse_args([])
        self.assertFalse(runner.USE_PAPER_EVENT_TEST_SETTINGS)
        self.assertIsNone(runner.PILOT_LEN_OVERRIDE)
        self.assertEqual(args.mc_times, 1000)
        self.assertEqual(asdict(runner.experiment_system(args)),
                         asdict(runner.build_system_config(runner.build_args())))
        self.assertEqual(runner.OUTPUT_ROOT.parent, runner.ROOT / "checkpoint")

    def test_low_load_defaults_are_aligned(self) -> None:
        from network.config import DEFAULT_CONFIG
        runner = self.runner
        cfg = runner.experiment_system(runner.parse_args([]))
        self.assertEqual((cfg.num_devices, cfg.num_antennas, cfg.pilot_len), (200, 32, 10))
        self.assertEqual((cfg.event_lambda, cfg.event_trigger_prob), (1.0, 0.5))
        self.assertEqual((cfg.event_sigma_m, cfg.background_activity_prob), (150.0, 0.005))
        self.assertEqual((cfg.cell_radius_m, cfg.noise_mode, cfg.snr_db), (500.0, "snr", 20.0))
        for other in (runner.SystemConfig(), runner.SystemConfig(**DEFAULT_CONFIG["network_train"]["system"])):
            self.assertEqual((other.event_lambda, other.event_trigger_prob), (1.0, 0.5))
            self.assertEqual(other.event_sigma_m, 150.0)
        self.assertEqual(runner.OUTPUT_ROOT.name, "covariance_261015_sigma150_trigger0p5_lambda1_mc1000_event")

    def test_shared_generator_uses_measured_receive_power(self) -> None:
        from dataclasses import replace
        import torch
        runner = self.runner
        cfg = replace(runner.experiment_system(runner.parse_args([])), noise_mode="snr")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            raw = runner.ActivityDataGenerator(cfg, torch.device("cpu")).sample_batch(3, return_raw=True)
        expected = ((raw["bh"].abs() ** 2).mean(dim=(1, 2)) * 10 ** (-cfg.snr_db / 10)).clamp_min(1e-30)
        torch.testing.assert_close(raw["noise_var"], expected, rtol=1e-6, atol=0)
        self.assertEqual(tuple(raw["s"].shape), (3, cfg.pilot_len, cfg.num_devices))

    def test_event_response_uses_configured_scale_and_background(self) -> None:
        from dataclasses import replace
        import torch
        runner = self.runner
        cfg = replace(runner.experiment_system(runner.parse_args([])), num_devices=3)
        generator = runner.ActivityDataGenerator(cfg, torch.device("cpu"))
        positions = torch.tensor([[[0.0, 0.0], [cfg.event_sigma_m, 0.0], [2 * cfg.event_sigma_m, 0.0]]])
        centers = torch.zeros(1, cfg.event_max_count, 2)
        with torch.random.fork_rng(devices=[]), patch.object(generator, "_sample_positions_m", return_value=centers):
            with patch.object(torch, "poisson", return_value=torch.ones(1)) as poisson:
                _, probabilities, _ = generator._sample_event_activity(positions)
                torch.testing.assert_close(poisson.call_args.args[0], torch.tensor([1.0]))
            expected = 0.005 + 0.995 * 0.5 * torch.exp(-torch.tensor([[0.0, 0.5, 2.0]]))
            torch.testing.assert_close(probabilities, expected)
            with patch.object(torch, "poisson", return_value=torch.zeros(1)):
                _, background, _ = generator._sample_event_activity(positions)
                torch.testing.assert_close(background, torch.full((1, 3), 0.005))
        old = runner.ActivityDataGenerator(replace(cfg, event_lambda=1.0, event_trigger_prob=0.9), torch.device("cpu"))
        torch.testing.assert_close(generator._correlation_from_positions(positions), old._correlation_from_positions(positions))

    def test_geometry_distance_scale_follows_event_scale(self) -> None:
        from dataclasses import replace
        import torch
        runner = self.runner
        cfg = replace(runner.experiment_system(runner.parse_args([])), num_devices=2)
        positions = torch.tensor([[[0.0, 0.0], [cfg.event_sigma_m, 0.0]]])
        generator = runner.ActivityDataGenerator(cfg, torch.device("cpu"))
        corr = generator._correlation_from_positions(positions)
        torch.testing.assert_close(corr[0, 0, 1], torch.exp(torch.tensor(-0.25)))
        torch.testing.assert_close(corr[0].diagonal(), torch.zeros(2))
        wider = runner.ActivityDataGenerator(replace(cfg, event_sigma_m=2 * cfg.event_sigma_m), torch.device("cpu"))
        self.assertLess(float(corr[0, 0, 1]), float(wider._correlation_from_positions(positions)[0, 0, 1]))

    def test_explicit_command_line_settings_take_priority(self) -> None:
        runner = self.runner
        with patch.object(runner, "USE_PAPER_EVENT_TEST_SETTINGS", True), patch.object(runner, "PILOT_LEN_OVERRIDE", None):
            args = runner.parse_args(["--num-users", "150", "--pilot-len", "12", "--cell-radius-m", "300",
                                      "--noise-mode", "snr", "--activity-mode", "independent"])
            cfg = runner.experiment_system(args)
        self.assertEqual((cfg.num_devices, cfg.pilot_len, cfg.cell_radius_m), (150, 12, 300.0))
        self.assertEqual((cfg.noise_mode, cfg.activity_mode), ("snr", "independent"))

    def test_independent_reference_keeps_paper_defaults_and_separate_output(self) -> None:
        from covariance_based.run_independent_reference import PaperSystem, REFERENCE_OUTPUT_DIR, experiment_system, parse_args
        runner = self.runner
        for preset_enabled in (False, True):
            with self.subTest(preset_enabled=preset_enabled), patch.object(runner, "USE_PAPER_EVENT_TEST_SETTINGS", preset_enabled), \
                    patch.object(runner, "PAPER_EVENT_TEST_SETTINGS", {"num_devices": 200, "pilot_len": 30}):
                args = parse_args([])
                cfg = experiment_system(args)
                self.assertEqual(cfg, PaperSystem())
                self.assertEqual((cfg.activity_mode, cfg.activity_prob), ("independent", 0.1))
                self.assertEqual(args.out_dir, REFERENCE_OUTPUT_DIR)
                self.assertNotEqual(args.out_dir, runner.PAPER_EVENT_TEST_OUTPUT_ROOT / "single")

    def test_independent_reference_explicit_arguments_take_priority(self) -> None:
        from covariance_based.run_independent_reference import experiment_system, parse_args
        args = parse_args(["--mc-times", "3", "--pilot-len", "12", "--out-dir", "/tmp/reference_override"])
        cfg = experiment_system(args)
        self.assertEqual((args.mc_times, cfg.pilot_len), (3, 12))
        self.assertEqual(args.out_dir, Path("/tmp/reference_override"))
        self.assertEqual((cfg.activity_mode, cfg.activity_prob), ("independent", 0.1))

    def test_snr_sweep_requires_snr_dependent_noise(self) -> None:
        runner = self.runner
        with patch.object(runner, "USE_PAPER_EVENT_TEST_SETTINGS", True), patch.object(runner, "PILOT_LEN_OVERRIDE", None):
            args = runner.parse_args(["--sweep", "SNR"])
            with self.assertRaisesRegex(ValueError, "fixed thermal noise"):
                runner.experiment_points(runner.experiment_system(args), args)
            args = runner.parse_args(["--sweep", "SNR", "--noise-mode", "snr"])
            points = runner.experiment_points(runner.experiment_system(args), args)
        self.assertEqual(len(points), 6)
        self.assertTrue(all(cfg.noise_mode == "snr" for _, _, cfg in points))


@unittest.skipUnless(find_spec("torch"), "Activity statistics tests require torch.")
class ActivityCountStatsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import covariance_based.run_monte_carlo as runner
        cls.runner = runner

    def test_counts_histogram_quantiles_and_empty_trials(self) -> None:
        stats = self.runner.summarize_activity_counts([0, 1, 2, 2, 5], 5)
        self.assertEqual((stats["min_active"], stats["max_active"], stats["mean_active"]), (0, 5, 2.0))
        self.assertEqual(stats["histogram_by_active_count"], [1, 1, 2, 0, 0, 1])
        self.assertEqual((stats["zero_active_trials"], stats["nonempty_trials"], stats["min_active_nonempty"]), (1, 4, 1))
        self.assertEqual(stats["mean_active_nonempty"], 2.5)
        self.assertAlmostEqual(stats["p95_active"], 4.4)
        empty = self.runner.summarize_activity_counts([0, 0], 3)
        self.assertIsNone(empty["min_active_nonempty"])
        self.assertIsNone(empty["mean_active_nonempty"])
        for invalid in ([], [-1], [6], [0.5], [float("nan")], [[1]]):
            with self.assertRaises(ValueError):
                self.runner.summarize_activity_counts(invalid, 5)

    def test_stats_are_grouped_by_experiment_point(self) -> None:
        import json
        runner = self.runner
        metadata = {"method": "covariance", "sweep": "N", "sweep_value": "5", "N": 5, "M": 3,
                    "L": 2, "snr_db": 20.0, "activity_mode": "event", "noise_mode": "snr"}
        rows = [{**metadata, "active": value} for value in (0, 2, 3)]
        rows.append({**metadata, "sweep_value": "10", "N": 10, "active": 4})
        with TemporaryDirectory() as directory:
            out = Path(directory)
            runner.write_csv(out / "trial_results.csv", runner.POINT_FIELDS + ["active"], rows)
            records = runner.save_activity_count_stats(out)
            self.assertEqual(len(records), 2)
            self.assertEqual((records[0]["mc_times"], records[0]["max_active"]), (3, 3))
            self.assertEqual((records[1]["mc_times"], records[1]["mean_active"]), (1, 4.0))
            self.assertEqual(json.loads((out / "activity_count_stats.json").read_text()), records)


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
