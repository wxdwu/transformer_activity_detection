from __future__ import annotations

import json
import io
import runpy
from contextlib import redirect_stderr
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

import numpy as np
import torch

from experiments import run_pe
from experiments.pe_metrics import METHODS, METRIC, check_compatible, exact_min_pe, refresh_sweep, save_method_result
from network.data import ActivityDataGenerator
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
    def test_different_observations_flag_is_explicit_and_plot_only(self) -> None:
        args = run_pe.parse_args([], ("Covariance", "Box-NNLS"), "evaluate")
        self.assertFalse(args.allow_different_observations)
        args = run_pe.parse_args(["--stage", "plot", "--allow-different-observations"],
                                 ("Covariance", "Box-NNLS"), "evaluate")
        self.assertTrue(args.allow_different_observations)
        for stage in ("evaluate", "train-evaluate"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_pe.parse_args(["--stage", stage, "--allow-different-observations"],
                                  ("Covariance", "Box-NNLS"), "evaluate")

    def test_plot_only_forwards_observation_policy_without_training_or_evaluation(self) -> None:
        with TemporaryDirectory() as temporary, patch.object(run_pe, "refresh_sweep") as refresh, \
                patch.object(run_pe, "run_training") as train, \
                patch.object(run_pe, "evaluate_shared_trials") as evaluate:
            run_pe.main(["--stage", "plot", "--sweeps", "L", "--values", "4", "20",
                         "--out-root", temporary, "--allow-different-observations"])
            refresh.assert_called_once_with(Path(temporary) / "pe_vs_L", "L", "pilot_len",
                                            [(4, "L04"), (20, "L20")], "log",
                                            allow_different_observations=True)
            train.assert_not_called()
            evaluate.assert_not_called()

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

    def test_default_l_sweeps_stop_at_14_for_neural_and_traditional_runners(self) -> None:
        from covariance_based.run_monte_carlo import experiment_points, parse_args
        from network.run_varying_l import parse_pilot_lens

        expected = [4, 6, 8, 10, 12, 14]
        for raw in (None, "", "  "):
            self.assertEqual(parse_pilot_lens(raw), expected)
        self.assertEqual([value for value, _ in run_pe.sweep_points("L")], expected)
        cfg = run_pe.build_system_config(run_pe.build_args())
        points = experiment_points(cfg, parse_args(["--sweep", "L"]))
        self.assertEqual([point_cfg.pilot_len for _, _, point_cfg in points], expected)
        for stage in ("train-evaluate", "evaluate", "plot"):
            args = run_pe.parse_args(["--stage", stage], ("GI-HT", "HT"), "train-evaluate", fixed_sweep="L")
            self.assertEqual([value for value, _ in run_pe.sweep_points("L", args.values)], expected)
        self.assertEqual(run_pe.SWEEPS["N"][1], [50, 100, 150, 200, 250])
        self.assertEqual(run_pe.SWEEPS["SNR"][1], [10, 12, 14, 16, 18, 20])
        self.assertEqual(cfg.pilot_len, 10)

    def test_antenna_and_event_scale_sweeps_have_matching_defaults_and_labels(self) -> None:
        from covariance_based.run_monte_carlo import experiment_points, parse_args

        expected = {"M": [32, 48, 64, 80, 96, 112, 128],
                    "event_sigma_m": [60, 70, 80, 90, 100, 110, 120]}
        base = run_pe.build_system_config(run_pe.build_args())
        for sweep, values in expected.items():
            self.assertEqual(run_pe.SWEEPS[sweep][1], values)
            self.assertEqual([value for value, _ in run_pe.sweep_points(sweep)], values)
            points = experiment_points(base, parse_args(["--sweep", sweep]))
            self.assertEqual([getattr(cfg, run_pe.SWEEPS[sweep][0]) for _, _, cfg in points], values)
        self.assertEqual(run_pe.sweep_points("M", [128, 32]), [(32, "M032"), (128, "M128")])
        self.assertEqual(run_pe.sweep_points("event_sigma_m", [120, 60.5]),
                         [(60.5, "SIGMA60p5"), (120.0, "SIGMA120")])
        fractional = experiment_points(base, parse_args(["--sweep", "event_sigma_m", "--values", "60.5"]))
        self.assertEqual(fractional[0][2].event_sigma_m, 60.5)
        for sweep, values in (("M", [32.5]), ("M", [0]), ("event_sigma_m", [0]),
                              ("event_sigma_m", [-1]), ("event_sigma_m", [float("inf")]),
                              ("event_sigma_m", [60, 60])):
            with self.subTest(sweep=sweep, values=values), self.assertRaises(ValueError):
                run_pe.sweep_points(sweep, values)
        for sweep, values in (("M", [32.5]), ("event_sigma_m", [0])):
            with self.assertRaises(ValueError):
                experiment_points(base, parse_args(["--sweep", sweep, "--values", str(values[0])]))

    def test_new_neural_entry_points_select_only_their_own_sweep(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for script, sweep in (("run_pe_vs_M.py", "M"), ("run_pe_vs_event_sigma.py", "event_sigma_m")):
            with self.subTest(script=script), patch.object(run_pe, "main") as main:
                runpy.run_path(str(root / "network" / script), run_name="__main__")
                main.assert_called_once_with(default_methods=("GI-HT", "HT"), default_stage="train-evaluate",
                                             fixed_sweep=sweep)

    def test_antenna_sweep_changes_raw_shapes_without_changing_geometry(self) -> None:
        torch.set_num_threads(1)
        base = run_pe.build_system_config(run_pe.build_args())
        batches = []
        for antennas in (32, 128):
            torch.manual_seed(42)
            cfg = run_pe.replace(base, num_antennas=antennas)
            batch = ActivityDataGenerator(cfg, torch.device("cpu")).sample_batch(4, return_raw=True)
            self.assertEqual(batch["y"].shape, (4, base.pilot_len, antennas))
            self.assertEqual(batch["h"].shape, (4, base.num_devices, antennas))
            self.assertEqual(batch["x_y"].shape, (4, 2 * base.pilot_len**2))
            self.assertTrue(torch.isfinite(batch["y"]).all())
            batches.append(batch)
        for name in ("positions", "b", "corr_matrix", "activity_prob_map", "label"):
            torch.testing.assert_close(batches[0][name], batches[1][name], rtol=0, atol=0)

    def test_event_scale_sweep_updates_triggering_and_geometry_together(self) -> None:
        torch.set_num_threads(1)
        base = run_pe.build_system_config(run_pe.build_args())
        batches = []
        for sigma in (60.0, 120.0):
            torch.manual_seed(42)
            cfg = run_pe.replace(base, event_sigma_m=sigma)
            batches.append(ActivityDataGenerator(cfg, torch.device("cpu")).sample_batch(4, return_raw=True))
        for name in ("positions", "b", "event_centers"):
            torch.testing.assert_close(batches[0][name], batches[1][name], rtol=0, atol=0)
        for name in ("corr_matrix", "activity_prob_map"):
            self.assertTrue(torch.all(batches[0][name] <= batches[1][name]))
            self.assertTrue(torch.any(batches[0][name] < batches[1][name]))

    def test_loss_l_svg_axis_matches_shorter_default_range(self) -> None:
        from network.run_varying_l import write_loss_vs_l_svg

        rows = {"addcorr": [{"pilot_len": 4, "loss": 0.2}, {"pilot_len": 14, "loss": 0.1}]}
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "loss.svg"
            write_loss_vs_l_svg(path, rows)
            root = ElementTree.parse(path).getroot()
            namespace = {"svg": "http://www.w3.org/2000/svg"}
            ticks = [node.text for node in root.findall("svg:text", namespace) if node.get("y") == "504"]
            self.assertEqual(ticks, ["4", "6", "8", "10", "12", "14"])
            circles = root.findall("svg:circle", namespace)
            self.assertEqual([float(node.get("cx")) for node in circles], [90.0, 865.0])

    def test_training_scene_changes_only_the_swept_field_and_geometry_flags(self) -> None:
        cli = run_pe.parse_args([], ("GI-HT", "HT"), "train-evaluate")
        original = run_pe.build_system_config(run_pe.build_args())
        for sweep, value in (("L", 16), ("N", 250), ("SNR", 12.0), ("M", 128), ("event_sigma_m", 60.0)):
            for method in ("GI-HT", "HT"):
                args = run_pe.training_args(cli, sweep, value, Path("/tmp/pe_args"), method)
                expected = {**asdict(original), run_pe.SWEEPS[sweep][0]: value}
                self.assertEqual(asdict(run_pe.build_system_config(args)), expected)
                cfg = run_pe.build_model_config(args)
                self.assertEqual(cfg["signal_token_mode"], "covariance_rows")
                self.assertTrue(all(cfg[name] == (method == "GI-HT") for name in run_pe.GEOMETRY_FLAGS))

    def test_new_sweeps_save_four_method_summaries_and_combined_point_figures(self) -> None:
        base = run_pe.build_system_config(run_pe.build_args())
        labels = np.zeros((2, base.num_devices), dtype=np.uint8)
        labels[:, 0] = 1
        scores = np.full(labels.shape, 0.1)
        scores[:, 0] = 0.9
        scores[0, 1] = 0.95
        trials = [{"trial": 1}, {"trial": 2}]
        with TemporaryDirectory() as temporary:
            for sweep, values, xlabel in (("M", [32, 128], "Number of antennas M"),
                                           ("event_sigma_m", [60, 120], "Event influence scale")):
                field = run_pe.SWEEPS[sweep][0]
                points = run_pe.sweep_points(sweep, values)
                out = Path(temporary) / f"pe_vs_{sweep}"
                for value, label in points:
                    cfg = run_pe.replace(base, **{field: value})
                    for method in METHODS:
                        neural = method in ("GI-HT", "HT")
                        metadata = {"sweep": sweep, "value": value, "system_config": asdict(cfg), "mc_times": 2,
                                    "evaluation_seed": 42, "dataset_sha256": f"same-{label}",
                                    "checkpoint_epoch": 100 if neural else None,
                                    "inference_batch_size": 128 if neural else None}
                        save_method_result(out / "points" / label / method, method, scores, labels, metadata, trials)
                rows = refresh_sweep(out, sweep, field, points, "log")
                self.assertEqual(len(rows), 8)
                configuration = json.loads((out / "summary_config.json").read_text())
                self.assertEqual(configuration["requested_values"], values)
                for suffix in ("png", "svg", "pdf"):
                    self.assertTrue((out / f"pe_vs_{sweep}.{suffix}").is_file())
                    for _, label in points:
                        self.assertTrue((out / "points" / label / f"pm_pf_comparison.{suffix}").is_file())
                        for method in METHODS:
                            self.assertFalse((out / "points" / label / method / f"pm_pf_curve.{suffix}").exists())
                self.assertIn(xlabel, (out / f"pe_vs_{sweep}.svg").read_text())

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

    def test_same_scene_merge_relaxes_only_observation_hash_check(self) -> None:
        cfg = asdict(run_pe.build_system_config(run_pe.build_args()))
        first = {"metric": METRIC, "value": 10, "system_config": cfg, "mc_times": 1000,
                 "evaluation_seed": 42, "decision_rule": "score > threshold", "threshold_range": [0, 1],
                 "dataset_sha256": "local", "method": "GI-HT", "inference_batch_size": 128,
                 "checkpoint_epoch": 100}
        second = {**first, "method": "HT", "dataset_sha256": "server"}
        check_compatible([first, second], "pilot_len", allow_different_observations=True)
        for changes in ({"mc_times": 999}, {"evaluation_seed": 43}, {"decision_rule": "score >= threshold"},
                        {"threshold_range": [-1, 1]}, {"inference_batch_size": 4}, {"checkpoint_epoch": 99},
                        {"metric": "PM+PF/2"}, {"system_config": {**cfg, "event_trigger_prob": 0.5}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                check_compatible([first, {**second, **changes}], "pilot_len", allow_different_observations=True)

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
                    self.assertFalse((out / "points/L10" / method / f"pm_pf_curve.{suffix}").exists())
            rows = refresh_sweep(out, "L", "pilot_len", [(10, "L10")], "linear")
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(json.loads((out / "summary_config.json").read_text())["completed"]), 2)
            self.assertIn("Box-NNLS", (out / "pe_vs_L.svg").read_text())
            self.assertTrue((out / "points/L10/pm_pf_comparison.svg").is_file())
            configuration = json.loads((out / "summary_config.json").read_text())
            self.assertFalse(configuration["allow_different_observations"])
            self.assertTrue(configuration["observation_hashes_match"])
            path = out / "points/L10/Box-NNLS/metrics.json"
            changed_record = json.loads(path.read_text())
            changed_record["dataset_sha256"] = "different"
            path.write_text(json.dumps(changed_record), encoding="utf-8")
            original_metrics = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "identical observations"):
                refresh_sweep(out, "L", "pilot_len", [(10, "L10")], "linear")
            merged = refresh_sweep(out, "L", "pilot_len", [(10, "L10")], "linear",
                                   allow_different_observations=True)
            self.assertEqual(path.read_bytes(), original_metrics)
            self.assertEqual([r["min_pe"] for r in merged], [r["min_pe"] for r in rows])
            configuration = json.loads((out / "summary_config.json").read_text())
            self.assertTrue(configuration["allow_different_observations"])
            self.assertFalse(configuration["observation_hashes_match"])
            self.assertEqual(configuration["observation_hashes"]["L10"],
                             {"Covariance": "same", "Box-NNLS": "different"})

    def test_saving_each_method_keeps_data_without_calling_plotter(self) -> None:
        scores, labels = np.array([[0.6, 0.4]]), np.array([[1, 0]])
        with TemporaryDirectory() as temporary, patch("experiments.pe_metrics.plot_comparison") as plot:
            for method in METHODS:
                directory = Path(temporary) / method
                record = save_method_result(directory, method, scores, labels, {"mc_times": 1}, [{"trial": 1}])
                self.assertEqual(record["min_pe"], 0)
                self.assertEqual({p.name for p in directory.iterdir()},
                                 {"scores.npz", "exact_curve.npz", "pm_pf_curve.csv", "trials.csv", "metrics.json"})
                with np.load(directory / "scores.npz") as data:
                    np.testing.assert_array_equal(data["scores"], scores)
                    np.testing.assert_array_equal(data["labels"], labels)
            plot.assert_not_called()

    def test_single_method_point_skips_comparison_but_keeps_sweep_summary(self) -> None:
        cfg = asdict(run_pe.build_system_config(run_pe.build_args()))
        metadata = {"sweep": "L", "value": 10, "system_config": cfg, "mc_times": 1,
                    "evaluation_seed": 42, "dataset_sha256": "same", "checkpoint_epoch": None}
        with TemporaryDirectory() as temporary, patch("experiments.pe_metrics.plot_comparison") as comparison, \
                patch("experiments.pe_metrics.plot_pe") as summary:
            out = Path(temporary)
            save_method_result(out / "points/L10/Covariance", "Covariance",
                               np.array([[0.6, 0.4]]), np.array([[1, 0]]), metadata, [{"trial": 1}])
            rows = refresh_sweep(out, "L", "pilot_len", [(10, "L10")], "log")
            self.assertEqual(len(rows), 1)
            comparison.assert_not_called()
            summary.assert_called_once_with(out, "L", rows, "log")
            self.assertTrue((out / "pe_vs_L.csv").is_file())
            self.assertTrue((out / "summary_config.json").is_file())

    def test_interrupted_rewrite_does_not_publish_stale_optimal_metrics(self) -> None:
        metadata = {"mc_times": 1}
        scores, labels = np.array([[0.6, 0.4]]), np.array([[1, 0]])
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            save_method_result(directory, "HT", scores, labels, metadata, [{"trial": 1}])
            self.assertTrue((directory / "metrics.json").is_file())
            with patch("experiments.pe_metrics.write_json", side_effect=RuntimeError("interrupted save")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    save_method_result(directory, "HT", scores, labels, metadata, [{"trial": 1}])
            self.assertFalse((directory / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
