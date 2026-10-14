from __future__ import annotations

import csv
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from covariance_based.detector import covariance_mle_activity
from covariance_based.run_independent_reference import (
    MC_TIMES, PAPER_CASES, PaperSystem, direct_coordinate_check, experiment_system,
    main, paper_realization, parse_args, plot_reference, save_count_diagnostic, threshold_grid,
)


class PaperReferenceTests(unittest.TestCase):
    def test_paper_defaults_and_single_case_override(self) -> None:
        args = parse_args([])
        cfg = experiment_system(args)
        self.assertEqual(args.mc_times, 5000)
        self.assertEqual(MC_TIMES, 5000)
        self.assertEqual(tuple(args.cases), PAPER_CASES)
        self.assertEqual((cfg.num_devices, cfg.cell_radius_m, cfg.pmax_dbm), (100, 250, 23))
        self.assertAlmostEqual(cfg.noise_variance / 1e-13, 1.2589254117941663, places=12)
        self.assertEqual(parse_args(["--pilot-len", "10"]).cases, [(10, 32)])
        self.assertEqual(parse_args(["--num-antennas", "64"]).cases, [(8, 64)])
        for argv in (["--cases", "8:0"], ["--cases", "8:32", "8:32"],
                     ["--cases", "bad"], ["--cases", "8:32", "--pilot-len", "8"],
                     ["--pilot-len", "0"], ["--activity-prob", "1.1"],
                     ["--activity-mode", "event"], ["--noise-mode", "snr"],
                     ["--active-count", "101"], ["--active-count", "-1"],
                     ["--activity-mode", "fixed_count"], ["--num-users", "0", "--active-count", "0"],
                     ["--active-count", "10", "--activity-prob", "nan"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_generator_distribution_and_physical_identity(self) -> None:
        cfg = PaperSystem(num_devices=6000, pilot_len=8, num_antennas=8)
        raw = paper_realization(cfg, np.random.default_rng(2), 8, 8)
        self.assertEqual(raw["b"].dtype, np.complex128)
        self.assertAlmostEqual(float(np.mean(np.abs(raw["pilots"]) ** 2)), 1.0, delta=0.03)
        self.assertAlmostEqual(float(np.mean(np.abs(raw["channel"]) ** 2)), 1.0, delta=0.03)
        self.assertAlmostEqual(float(raw["labels"].mean()), 0.1, delta=0.015)
        radial_power = np.sum(raw["positions"] ** 2, axis=1) / cfg.cell_radius_m ** 2
        self.assertAlmostEqual(float(radial_power.mean()), 0.5, delta=0.015)
        self.assertTrue(np.all(radial_power <= 1.0))
        np.testing.assert_allclose(raw["power"] * raw["gain"],
                                  10 ** ((cfg.pmax_dbm - 30) / 10) * raw["gain"].min(), rtol=1e-14, atol=0)
        np.testing.assert_allclose(raw["y"], (raw["b"] * raw["labels"]) @ raw["channel"] + raw["noise"],
                                  rtol=1e-14, atol=0)
        cfg = PaperSystem(num_devices=5, pilot_len=600, num_antennas=64)
        raw = paper_realization(cfg, np.random.default_rng(3), 600, 64)
        self.assertAlmostEqual(float(np.mean(np.abs(raw["noise"]) ** 2)) / cfg.noise_variance, 1.0, delta=0.03)

    def test_cases_share_geometry_labels_and_signal_prefixes(self) -> None:
        short = paper_realization(PaperSystem(pilot_len=7), np.random.default_rng(4), 8, 64)
        long = paper_realization(PaperSystem(num_antennas=64), np.random.default_rng(4), 8, 64)
        for key in ("positions", "labels", "power", "gain"):
            np.testing.assert_array_equal(short[key], long[key])
        np.testing.assert_array_equal(short["b"], long["b"][:7])
        np.testing.assert_allclose(short["y"], long["y"][:7, :32], rtol=1e-14, atol=0)
        rng = np.random.default_rng(5)
        counts = [int(paper_realization(PaperSystem(), rng, 8, 64)["labels"].sum()) for _ in range(30)]
        self.assertGreater(len(set(counts)), 1)

    def test_fixed_count_is_explicit_and_preserves_physical_rng_pairing(self) -> None:
        args = parse_args(["--active-count", "10"])
        cfg = experiment_system(args)
        self.assertEqual((cfg.activity_mode, cfg.active_count, cfg.activity_prob), ("fixed_count", 10, 0.1))
        self.assertIn("fixedK10", str(args.out_dir))
        for count in (0, 10, 100):
            raw = paper_realization(replace(cfg, active_count=count), np.random.default_rng(12), 8, 64)
            self.assertEqual(int(raw["labels"].sum()), count)
        independent = paper_realization(PaperSystem(), np.random.default_rng(13), 8, 64)
        fixed = paper_realization(cfg, np.random.default_rng(13), 8, 64)
        for name in ("b", "pilots", "positions", "channel", "noise"):
            np.testing.assert_array_equal(independent[name], fixed[name])
        rng = np.random.default_rng(14)
        labels = np.stack([paper_realization(cfg, rng, 8, 64)["labels"] for _ in range(1000)])
        np.testing.assert_array_equal(labels.sum(axis=1), np.full(1000, 10))
        self.assertLess(float(np.max(np.abs(labels.mean(axis=0) - 0.1))), 0.04)

    def test_fast_solver_matches_direct_inversion(self) -> None:
        cfg = PaperSystem(num_devices=15, pilot_len=4, activity_prob=0.2)
        raw = paper_realization(cfg, np.random.default_rng(6), 4, 32)
        c = raw["y"] @ raw["y"].conj().T / cfg.num_antennas
        for random_order in (False, True):
            result = covariance_mle_activity(raw["b"], c, cfg.noise_variance, max_sweeps=500,
                tolerance=1e-7, random_order=random_order, rng=np.random.default_rng(7))
            direct, objective, step = direct_coordinate_check(raw["b"], c, cfg.noise_variance,
                result.sweeps, np.random.default_rng(7), random_order)
            self.assertTrue(result.converged)
            np.testing.assert_allclose(result.activities, direct, atol=1e-8, rtol=0)
            self.assertAlmostEqual(result.objective_history[-1], objective, places=8)
            self.assertLessEqual(step, 1.1e-7)

    def test_dense_grid_includes_exact_boundaries(self) -> None:
        thresholds = threshold_grid(101)
        self.assertEqual((thresholds[0], thresholds[-1]), (0.0, 1.0))
        self.assertTrue(np.all(np.diff(thresholds) > 0))
        self.assertLess(thresholds[1], 1e-7)
        self.assertGreater(thresholds[-2], 1 - 1e-7)

    def test_small_run_records_pooled_counts_scores_and_audit(self) -> None:
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            main(["--mc-times", "3", "--cases", "4:8", "5:8", "--num-users", "10",
                  "--activity-prob", "0.4", "--audit-trials", "2", "--num-thresholds", "11",
                  "--out-dir", directory, "--no-plot", "--log-every", "0"])
            with (Path(directory) / "summary.csv").open() as handle:
                summaries = list(csv.DictReader(handle))
            self.assertEqual(len(summaries), 2)
            for summary in summaries:
                case = Path(directory) / f"L{summary['L']}_M{summary['M']}"
                with np.load(case / "activity_scores.npz") as data:
                    scores, labels = data["scores"], data["labels"]
                self.assertEqual(scores.shape, (3, 10))
                self.assertEqual(int(summary["active"]), int(labels.sum()))
                self.assertEqual(int(summary["inactive"]), int((~labels).sum()))
                self.assertEqual(int(summary["missed"]), int(np.count_nonzero(labels & (scores <= 0.5))))
                self.assertEqual(int(summary["false_alarms"]), int(np.count_nonzero(~labels & (scores > 0.5))))
                self.assertEqual(int(summary["active_at_zero"]), int(np.count_nonzero(labels & (scores == 0))))
                with (case / "solver_audit.csv").open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_three_curve_plot_uses_paper_axes_without_fake_probabilities(self) -> None:
        summaries = [{"L": l, "M": m} for l, m in PAPER_CASES]
        curves = [[{"PM": 0.0, "PF": 0.1}, {"PM": 0.1, "PF": 0.01}, {"PM": 1.0, "PF": 0.0}]] * 3
        with TemporaryDirectory() as directory, patch("matplotlib.figure.Figure.savefig", autospec=True) as save:
            plot_reference(Path(directory), summaries, curves)
        ax = save.call_args_list[0].args[0].axes[0]
        self.assertEqual((ax.get_xscale(), ax.get_yscale()), ("log", "log"))
        self.assertEqual(len(ax.lines), 3)
        self.assertEqual([line.get_linestyle() for line in ax.lines], ["-", "--", ":"])
        self.assertTrue(np.isnan(ax.lines[0].get_xdata()[0]))
        self.assertTrue(np.isnan(ax.lines[0].get_ydata()[-1]))

    def test_count_diagnostic_selects_only_requested_support_size(self) -> None:
        scores = np.array([[0.8, 0.1, 0.9, 0.2], [0.1, 0.8, 0.2, 0.7], [0.1, 0.1, 0.1, 0.9]])
        labels = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]], dtype=bool)
        summary = {"method": "covariance_mle_cd", "sweep": "paper_L_M", "sweep_value": "L3_M8",
                   "N": 4, "M": 8, "L": 3, "snr_db": "", "activity_mode": "independent",
                   "noise_mode": "thermal", "mc_times": 3, "threshold": 0.5}
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / "L3_M8"
            path.mkdir()
            np.savez_compressed(path / "activity_scores.npz", scores=scores, labels=labels)
            summaries, curves = save_count_diagnostic(Path(directory), [summary], 2, 11, False)
            self.assertEqual(summaries[0]["selected_trials"], 2)
            self.assertEqual(summaries[0]["source_mc_times"], 3)
            self.assertEqual((summaries[0]["PM"], summaries[0]["PF"]), (0.25, 0.25))
            self.assertEqual((curves[0][0]["active"], curves[0][0]["inactive"]), (4, 4))


if __name__ == "__main__":
    unittest.main()
