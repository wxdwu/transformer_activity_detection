from __future__ import annotations

import json
import io
from contextlib import redirect_stderr
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import run_pe
from experiments.pe_metrics import METRIC, check_compatible, exact_min_pe, refresh_sweep, save_method_result
from network.run_pm_pf_comparison import evaluate_shared_trials


class ThresholdSelectionTests(unittest.TestCase):
    def test_exact_search_finds_optimum_between_coarse_grid_points(self) -> None:
        curve, best = exact_min_pe(np.array([0.5001, 0.50005]), np.array([1, 0]))
        self.assertEqual(best["min_pe"], 0)
        self.assertEqual(best["optimal_threshold"], 0.50005)
        self.assertTrue(np.all(np.diff(curve["PM"]) >= 0))
        self.assertTrue(np.all(np.diff(curve["PF"]) <= 0))

    def test_tied_scores_are_grouped_and_lowest_minimizer_is_selected(self) -> None:
        _, best = exact_min_pe(np.array([0, 0, 1, 1]), np.array([1, 0, 1, 0]))
        self.assertEqual(best["min_pe"], 1)
        self.assertEqual(best["optimal_threshold"], 0)
        self.assertEqual((best["PM"], best["PF"]), (0.5, 0.5))

    def test_complete_curve_and_minimum_match_independent_direct_counts(self) -> None:
        rng = np.random.default_rng(9)
        labels = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]])
        for _ in range(30):
            scores = rng.integers(0, 7, size=labels.shape) / 6
            curve, best = exact_min_pe(scores, labels)
            errors = []
            for index, threshold in enumerate(curve["threshold"]):
                predicted = scores > threshold
                missed = int(((~predicted) & labels.astype(bool)).sum())
                false = int((predicted & (~labels.astype(bool))).sum())
                error = missed / labels.sum() + false / (labels.size - labels.sum())
                self.assertEqual(int(curve["missed"][index]), missed)
                self.assertEqual(int(curve["false_alarms"][index]), false)
                self.assertAlmostEqual(float(curve["Pe"][index]), error)
                errors.append(error)
            self.assertAlmostEqual(best["min_pe"], min(errors))
            self.assertAlmostEqual(best["min_pe"], best["PM"] + best["PF"])
            self.assertLessEqual(best["min_pe"], 1)

    def test_sum_of_conditional_errors_is_not_total_device_error(self) -> None:
        _, best = exact_min_pe(np.array([0.9, 0.95, 0.1, 0.1, 0.1]), np.array([1, 0, 0, 0, 0]))
        self.assertEqual(best["min_pe"], 0.25)
        self.assertNotEqual(best["min_pe"], (best["missed"] + best["false_alarms"]) / 5)

    def test_invalid_or_undefined_metrics_are_not_replaced_with_zero(self) -> None:
        for scores, labels in (([], []), ([0.2], [0]), ([0.2], [1]), ([np.nan, 0.3], [1, 0]),
                               ([-0.1, 0.3], [1, 0]), ([1.1, 0.2], [1, 0]), ([0.1, 0.3], [2, 0])):
            with self.assertRaises(ValueError):
                exact_min_pe(np.array(scores), np.array(labels))


class PeExperimentTests(unittest.TestCase):
    def test_fixed_entry_points_are_restricted_to_one_sweep(self) -> None:
        for sweep in run_pe.SWEEPS:
            args = run_pe.parse_args([], ("GI-HT", "HT"), "train-evaluate", fixed_sweep=sweep)
            self.assertEqual(args.sweeps, [sweep])
            self.assertEqual(args.methods, ["GI-HT", "HT"])
            self.assertEqual(args.stage, "train-evaluate")
            self.assertEqual(args.epochs, 100)
            other = "N" if sweep == "L" else "L"
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_pe.parse_args(["--sweeps", other], ("GI-HT", "HT"), "train-evaluate", fixed_sweep=sweep)

    def test_sweeps_and_output_root_match_existing_experiments(self) -> None:
        from covariance_based.run_monte_carlo import DEFAULT_SWEEPS

        self.assertEqual(run_pe.SWEEPS, DEFAULT_SWEEPS)
        args = run_pe.parse_args([], ("Covariance", "Box-NNLS"), "evaluate")
        self.assertEqual(args.mc_times, 1000)
        self.assertEqual(args.out_root.name, "result_261015")
        self.assertEqual(args.epochs, 100)
        self.assertEqual(run_pe.sweep_points("L", [16, 4]), [(4, "L04"), (16, "L16")])
        self.assertEqual(run_pe.sweep_points("SNR", [12.5]), [(12.5, "SNR12p5")])
        for values in ([4, 4], [np.nan], [-2], [2.5]):
            with self.assertRaises(ValueError):
                run_pe.sweep_points("L", values)

    def test_training_scene_changes_only_the_swept_field_and_geometry_flags(self) -> None:
        cli = run_pe.parse_args([], ("GI-HT", "HT"), "train-evaluate")
        original = run_pe.build_system_config(run_pe.build_args())
        for sweep, value in (("L", 16), ("N", 250), ("SNR", 12.0)):
            for method in ("GI-HT", "HT"):
                args = run_pe.training_args(cli, sweep, value, Path("/tmp/pe_args"), method)
                expected = {**asdict(original), run_pe.SWEEPS[sweep][0]: value}
                self.assertEqual(asdict(run_pe.build_system_config(args)), expected)
                cfg = run_pe.build_model_config(args)
                self.assertEqual(cfg["signal_token_mode"], "covariance_rows")
                self.assertTrue(all(cfg[name] == (method == "GI-HT") for name in run_pe.GEOMETRY_FLAGS))

    def test_method_subsets_preserve_identical_observations_and_all_scores(self) -> None:
        class Detector(torch.nn.Module):
            def __init__(self, expects_geometry):
                super().__init__()
                self.expects_geometry = expects_geometry

            def forward(self, x_b, x_y, geometry):
                if (geometry is not None) != self.expects_geometry:
                    raise AssertionError("Geometry input disagrees with the selected method.")
                scores = torch.sigmoid(x_b.mean(dim=-1))
                return torch.logit(scores), scores

        torch.set_num_threads(1)
        models = {"GI-HT": Detector(True), "HT": Detector(False)}
        cfg = run_pe.build_system_config(run_pe.build_args())
        cfg = run_pe.replace(cfg, num_devices=8, num_antennas=4, pilot_len=3)
        args = SimpleNamespace(seed=42, mc_times=5, batch_size=3, log_every=0, max_sweeps=100,
                               tolerance=1e-4, include_nnls=True, nnls_max_iter=1000, nnls_tolerance=1e-10)
        full, labels, diagnostics, digest = evaluate_shared_trials(models, cfg, args, torch.device("cpu"))
        for methods in (("Covariance", "Box-NNLS"), ("GI-HT", "HT"), ("HT",), ("Box-NNLS",)):
            selected_models = {m: models[m] for m in methods if m in models}
            scores, other_labels, _, other_digest = evaluate_shared_trials(selected_models, cfg, args, torch.device("cpu"), methods=methods)
            self.assertEqual(set(scores), set(methods))
            np.testing.assert_array_equal(labels, other_labels)
            self.assertEqual(digest, other_digest)
            for method in methods:
                np.testing.assert_array_equal(scores[method], full[method])

    def test_completed_checkpoint_reuse_checks_epoch_and_training_settings(self) -> None:
        cli = run_pe.parse_args([], ("GI-HT",), "train-evaluate")
        cfg = run_pe.build_system_config(run_pe.build_args())
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            out = directory / "checkpoints/addcorr/L10"
            out.mkdir(parents=True)
            (out / "last.pt").touch()
            args = run_pe.training_args(cli, "L", 10, out, "GI-HT")
            metadata = {"system_config": asdict(cfg), "model_config": run_pe.build_model_config(args),
                        "epoch": 100, "training_config": dict(vars(args)), "training_batch_size": args.batch_size}
            with patch.object(run_pe, "load_detector", return_value=(torch.nn.Identity(), metadata)), \
                    patch.object(run_pe, "run_training") as train:
                run_pe.prepare_models(cli, "L", 10, "L10", directory, cfg, torch.device("cpu"))
                train.assert_not_called()
                metadata["epoch"] = 99
                with self.assertRaisesRegex(ValueError, "epoch"):
                    run_pe.prepare_models(cli, "L", 10, "L10", directory, cfg, torch.device("cpu"))
                metadata["epoch"] = 100
                metadata["training_config"]["steps_per_epoch"] = 2
                with self.assertRaisesRegex(ValueError, "steps_per_epoch"):
                    run_pe.prepare_models(cli, "L", 10, "L10", directory, cfg, torch.device("cpu"))
                train.assert_not_called()

    def test_merging_rejects_different_scenes_counts_hashes_and_neural_batches(self) -> None:
        cfg = asdict(run_pe.build_system_config(run_pe.build_args()))
        first = {"metric": METRIC, "value": 10, "system_config": cfg, "mc_times": 1000,
                 "evaluation_seed": 42, "decision_rule": "score > threshold", "threshold_range": [0, 1],
                 "dataset_sha256": "same", "method": "GI-HT", "inference_batch_size": 128, "checkpoint_epoch": 100}
        second = {**first, "method": "HT"}
        check_compatible([first, second], "pilot_len")
        for changes in ({"mc_times": 999}, {"dataset_sha256": "different"}, {"inference_batch_size": 4},
                        {"checkpoint_epoch": 99}, {"metric": "PM+PF/2"},
                        {"system_config": {**cfg, "event_trigger_prob": 0.5}}):
            with self.assertRaises(ValueError):
                check_compatible([first, {**second, **changes}], "pilot_len")

    def test_method_artifacts_and_combined_plot_preserve_exact_optimum(self) -> None:
        cfg = asdict(run_pe.build_system_config(run_pe.build_args()))
        metadata = {"sweep": "L", "value": 10, "system_config": cfg, "mc_times": 2,
                    "evaluation_seed": 42, "dataset_sha256": "same", "checkpoint_epoch": None}
        scores, labels = np.array([[0.5001, 0.50005], [0.4, 0.6]]), np.array([[1, 0], [1, 0]])
        trials = [{"trial": 1, "converged": 1}, {"trial": 2, "converged": 0}]
        with TemporaryDirectory() as temporary:
            out = Path(temporary)
            for method in ("Covariance", "Box-NNLS"):
                record = save_method_result(out / "points/L10" / method, method, scores, labels, metadata, trials)
                self.assertEqual(record["converged_fraction"], 0.5)
                with np.load(out / "points/L10" / method / "exact_curve.npz") as data:
                    self.assertAlmostEqual(float(data["Pe"].min()), record["min_pe"])
                for suffix in ("png", "svg", "pdf"):
                    self.assertTrue((out / "points/L10" / method / f"pm_pf_curve.{suffix}").is_file())
            rows = refresh_sweep(out, "L", "pilot_len", [(10, "L10")], "linear")
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(json.loads((out / "summary_config.json").read_text())["completed"]), 2)
            self.assertIn("Box-NNLS", (out / "pe_vs_L.svg").read_text())
            self.assertTrue((out / "points/L10/pm_pf_comparison.svg").is_file())

    def test_interrupted_rewrite_does_not_publish_stale_optimal_metrics(self) -> None:
        metadata = {"mc_times": 1}
        scores, labels = np.array([[0.6, 0.4]]), np.array([[1, 0]])
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            save_method_result(directory, "HT", scores, labels, metadata, [{"trial": 1}])
            self.assertTrue((directory / "metrics.json").is_file())
            with patch("experiments.pe_metrics.plot_comparison", side_effect=RuntimeError("interrupted plot")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    save_method_result(directory, "HT", scores, labels, metadata, [{"trial": 1}])
            self.assertFalse((directory / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
