from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from network import run_pm_pf_comparison as runner

import numpy as np
import torch


class PMPFComparisonTests(unittest.TestCase):
    def test_curve_matches_strict_threshold_counts(self) -> None:
        scores = np.array([[0, 0.5, 1, 0.5], [0.2, 0.8, 0.9, 0.0]])
        labels = np.array([[1, 1, 0, 0], [1, 0, 1, 0]])
        thresholds = np.array([0.0, 0.2, 0.5, 0.9, 1.0])
        curve = runner.curve_from_scores(scores, labels, thresholds, "test")
        for row, threshold in zip(curve, thresholds):
            predicted = scores > threshold
            self.assertEqual(row["missed"], int(np.sum(~predicted & labels.astype(bool))))
            self.assertEqual(row["false_alarms"], int(np.sum(predicted & ~labels.astype(bool))))
        self.assertEqual(curve[-1]["PM"], 1.0)
        self.assertEqual(curve[-1]["PF"], 0.0)
        self.assertTrue(all(a["PM"] <= b["PM"] and a["PF"] >= b["PF"] for a, b in zip(curve, curve[1:])))

    def test_invalid_scores_and_undefined_denominators(self) -> None:
        for scores, labels in ((np.array([np.nan]), np.array([1])),
                               (np.array([1.1]), np.array([1])),
                               (np.array([0.1]), np.array([2]))):
            with self.assertRaises(ValueError):
                runner.curve_from_scores(scores, labels, np.array([0.5]), "test")
        row = runner.curve_from_scores(np.array([0.2]), np.array([0]), np.array([0.5]), "test")[0]
        self.assertTrue(np.isnan(row["PM"]))
        self.assertEqual(row["PF"], 0.0)

    def test_pair_requires_same_scene_epoch_and_architecture(self) -> None:
        cfg = runner.SystemConfig(num_devices=3, pilot_len=2)
        first = {"system_config": runner.asdict(cfg), "epoch": 100, "training_batch_size": 128, "model_config": {"dim": 8}}
        second = {**first, "model_config": {"dim": 8, **dict.fromkeys(runner.GEOMETRY_FLAGS, False)}}
        self.assertEqual(runner.validate_pair(first, second), cfg)
        for change in ({"epoch": 90}, {"training_batch_size": 16}, {"system_config": {**runner.asdict(cfg), "num_devices": 4}},
                       {"model_config": {"dim": 16}}):
            with self.assertRaises(ValueError):
                runner.validate_pair(first, {**second, **change})

    def test_bad_checkpoint_is_reported_without_modifying_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            path.write_bytes(b"PK\x03\x04incomplete archive")
            with self.assertRaisesRegex(RuntimeError, "complete checkpoint"):
                runner.load_detector(path, "GI-HT", torch.device("cpu"))
            self.assertEqual(path.read_bytes(), b"PK\x03\x04incomplete archive")

    def test_shared_trials_do_not_depend_on_inference_batch_size(self) -> None:
        class Detector(torch.nn.Module):
            def __init__(self, requires_geometry: bool) -> None:
                super().__init__()
                self.requires_geometry = requires_geometry

            def forward(self, x_b, x_y, geometry):
                if self.requires_geometry != (geometry is not None):
                    raise AssertionError("HT must not receive geometry.")
                probs = torch.sigmoid(x_b.mean(dim=-1))
                return torch.logit(probs), probs

        cfg = runner.SystemConfig(num_devices=5, num_antennas=3, pilot_len=2)
        models = {"GI-HT": Detector(True), "HT": Detector(False)}

        results = []
        torch.set_num_threads(1)
        for size in (1, 3):
            args = runner.parse_args(["--mc-times", "4", "--batch-size", str(size), "--log-every", "0"])
            results.append(runner.evaluate_shared_trials(models, cfg, args, torch.device("cpu")))
        scores, labels, diagnostics, digest = results[0]
        other_scores, other_labels, other_diagnostics, other_digest = results[1]
        np.testing.assert_array_equal(labels, other_labels)
        self.assertEqual(digest, other_digest)
        self.assertEqual(diagnostics, other_diagnostics)
        for method in runner.METHODS:
            np.testing.assert_array_equal(scores[method], other_scores[method])

    def test_plot_contains_three_methods(self) -> None:
        scores = np.array([0, 0.2, 0.8, 1])
        labels = np.array([1, 0, 1, 0])
        curves = {m: runner.curve_from_scores(scores, labels, np.linspace(0, 1, 11), m) for m in runner.METHODS}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            runner.plot_comparison(path, curves)
            for suffix in ("png", "svg", "pdf"):
                self.assertGreater((path / f"pm_pf_comparison.{suffix}").stat().st_size, 1000)
            svg = (path / "pm_pf_comparison.svg").read_text()
            for method in runner.METHODS:
                self.assertIn(method, svg)

    def test_default_batch_size_inherits_checkpoint(self) -> None:
        self.assertIsNone(runner.parse_args([]).batch_size)
        from network import run_loss_vs_epoch
        from unittest.mock import patch

        with patch("sys.argv", ["run_loss_vs_epoch.py"]):
            training = run_loss_vs_epoch.parse_args()
        comparison = runner.parse_args([])
        self.assertEqual(comparison.giht_ckpt, training.out_dir / "addcorr" / "last.pt")
        self.assertEqual(comparison.ht_ckpt, training.out_dir / "noaddcorr" / "last.pt")

    def test_nnls_is_optional_and_defaults_are_explicit(self) -> None:
        self.assertFalse(runner.parse_args([]).include_nnls)
        args = runner.parse_args(["--include-nnls"])
        self.assertTrue(args.include_nnls)
        self.assertEqual(args.nnls_max_iter, 1000)
        self.assertEqual(args.nnls_tolerance, 1e-10)
        self.assertEqual(runner.parse_args([]).out_dir, runner.OUTPUT_DIR)
        self.assertNotEqual(args.out_dir, runner.OUTPUT_DIR)

    def test_nnls_preserves_original_shared_data_scores_and_mle_diagnostics(self) -> None:
        class Detector(torch.nn.Module):
            def forward(self, x_b, x_y, geometry):
                probabilities = torch.sigmoid(x_b.mean(dim=-1))
                return torch.logit(probabilities), probabilities

        models = {"GI-HT": Detector(), "HT": Detector()}
        cfg = runner.SystemConfig(num_devices=6, num_antennas=4, pilot_len=3)
        torch.set_num_threads(1)
        args = runner.parse_args(["--mc-times", "5", "--batch-size", "2", "--log-every", "0"])
        scores, labels, diagnostics, digest = runner.evaluate_shared_trials(models, cfg, args, torch.device("cpu"))
        args.include_nnls = True
        other_scores, other_labels, other_diagnostics, other_digest = runner.evaluate_shared_trials(models, cfg, args, torch.device("cpu"))
        np.testing.assert_array_equal(labels, other_labels)
        self.assertEqual(digest, other_digest)
        self.assertEqual(diagnostics, [{key: val for key, val in row.items() if not key.startswith("nnls_")} for row in other_diagnostics])
        for method in runner.METHODS:
            np.testing.assert_array_equal(scores[method], other_scores[method])
        nnls = other_scores[runner.NNLS_METHOD]
        self.assertTrue(np.isfinite(nnls).all())
        self.assertTrue(np.all((nnls >= 0) & (nnls <= 1)))
        self.assertTrue(all(row["nnls_converged"] for row in other_diagnostics))

    def test_plot_contains_four_methods_with_nnls(self) -> None:
        methods = (*runner.METHODS, runner.NNLS_METHOD)
        curves = {method: runner.curve_from_scores(np.array([0, 0.2, 0.8, 1]), np.array([1, 0, 1, 0]),
                                                 np.linspace(0, 1, 11), method) for method in methods}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            runner.plot_comparison(path, curves)
            svg = (path / "pm_pf_comparison.svg").read_text()
            for method in methods:
                self.assertIn(method, svg)

    def test_formal_comparison_count_is_separate_from_covariance_test(self) -> None:
        from covariance_based.run_monte_carlo import parse_args

        self.assertEqual(runner.parse_args([]).mc_times, 5000)
        self.assertEqual(parse_args([]).mc_times, 1000)

    def test_token_batch_norm_remains_batch_dependent_in_eval(self) -> None:
        from network.model import TokenBatchNorm

        norm = TokenBatchNorm(1).eval()
        sample = torch.tensor([[[0.0], [1.0]]])
        others = torch.tensor([[[10.0], [11.0]]])
        alone = norm(sample)
        grouped = norm(torch.cat([sample, others]))[:1]
        self.assertFalse(torch.allclose(alone, grouped))


if __name__ == "__main__":
    unittest.main()
