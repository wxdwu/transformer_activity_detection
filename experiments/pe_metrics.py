from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from covariance_based.run_monte_carlo import write_csv
from network.run_pm_pf_comparison import plot_comparison


METHODS = ("GI-HT", "HT", "Covariance", "Box-NNLS")
METRIC = "min_threshold(PM + PF)"
SUMMARY_FIELDS = ["method", "sweep", "value", "min_pe", "optimal_threshold", "PM", "PF",
                  "missed", "false_alarms", "active", "inactive", "mc_times", "checkpoint_epoch",
                  "converged_fraction", "dataset_sha256"]
CURVE_FIELDS = ["method", "threshold", "PM", "PF", "Pe", "missed", "false_alarms", "active", "inactive"]


def exact_min_pe(scores: np.ndarray, labels: np.ndarray) -> tuple[dict, dict]:
    scores, labels = np.asarray(scores), np.asarray(labels)
    if scores.shape != labels.shape or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Nonempty finite scores must have the same shape as labels.")
    if np.any((scores < 0) | (scores > 1)) or np.any((labels != 0) & (labels != 1)):
        raise ValueError("Scores must lie in [0,1] and labels must be binary.")
    active_mask = labels.astype(bool)
    active, inactive = int(active_mask.sum()), int((~active_mask).sum())
    if not active or not inactive:
        raise ValueError("PM+PF requires both active and inactive instances; increase mc-times.")
    thresholds = np.unique(np.concatenate(([0.0], scores.ravel().astype(np.float64), [1.0])))
    missed = np.searchsorted(np.sort(scores[active_mask]), thresholds, side="right")
    false_alarms = inactive - np.searchsorted(np.sort(scores[~active_mask]), thresholds, side="right")
    pm, pf = missed / active, false_alarms / inactive
    # Integer counts give exact tie handling; sorted thresholds select the lowest minimizer.
    best = int(np.argmin(missed * inactive + false_alarms * active))
    curve = {"threshold": thresholds, "PM": pm, "PF": pf, "Pe": pm + pf,
             "missed": missed, "false_alarms": false_alarms}
    optimum = {"min_pe": float(pm[best] + pf[best]), "optimal_threshold": float(thresholds[best]),
               "PM": float(pm[best]), "PF": float(pf[best]), "missed": int(missed[best]),
               "false_alarms": int(false_alarms[best]), "active": active, "inactive": inactive}
    return curve, optimum


def sampled_curve(curve: dict, optimum: dict, method: str, limit: int = 2001) -> list[dict]:
    count = len(curve["threshold"])
    best = int(np.searchsorted(curve["threshold"], optimum["optimal_threshold"]))
    indices = np.unique(np.append(np.linspace(0, count - 1, min(limit, count), dtype=int), best))
    return [{"method": method, "active": optimum["active"], "inactive": optimum["inactive"],
             **{key: value[index].item() for key, value in curve.items()}} for index in indices]


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_method_result(out: Path, method: str, scores: np.ndarray, labels: np.ndarray,
                       metadata: dict, trials: list[dict]) -> dict:
    curve, optimum = exact_min_pe(scores, labels)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").unlink(missing_ok=True)
    np.savez_compressed(out / "scores.npz", scores=scores, labels=labels)
    np.savez_compressed(out / "exact_curve.npz", **curve)
    rows = sampled_curve(curve, optimum, method)
    write_csv(out / "pm_pf_curve.csv", CURVE_FIELDS, rows)
    write_csv(out / "trials.csv", list(trials[0]), trials)
    plot_comparison(out, {method: rows}, stem="pm_pf_curve")
    record = {**metadata, **optimum, "schema_version": 1, "method": method, "metric": METRIC,
              "decision_rule": "score > threshold", "threshold_range": [0.0, 1.0],
              "threshold_selection": "one empirical optimum per method and parameter point, all score breakpoints",
              "tie_break": "lowest minimizing threshold", "curve_csv": "sampled exact breakpoints plus optimum",
              "converged_fraction": (sum(int(t["converged"]) for t in trials) / len(trials)
                                     if "converged" in trials[0] else None)}
    # Publish the record after all artifacts so interrupted points are not aggregated.
    write_json(out / "metrics.json", record)
    return record


def check_compatible(records: list[dict], sweep_field: str) -> None:
    if not records:
        return
    reference = records[0]
    base = {k: v for k, v in reference["system_config"].items() if k != sweep_field}
    for record in records:
        other_base = {k: v for k, v in record["system_config"].items() if k != sweep_field}
        if record["metric"] != METRIC or other_base != base:
            raise ValueError("Cannot merge results with different metrics or unswept scene settings.")
        for name in ("mc_times", "evaluation_seed", "decision_rule", "threshold_range"):
            if record[name] != reference[name]:
                raise ValueError(f"Cannot merge results with different {name}.")
        same_point = [r for r in records if r["value"] == record["value"]]
        if len({r["dataset_sha256"] for r in same_point}) != 1:
            raise ValueError("Methods at the same point must use identical observations.")
        neural = [r for r in same_point if r["method"] in ("GI-HT", "HT")]
        if len({(r["inference_batch_size"], r["checkpoint_epoch"]) for r in neural}) > 1:
            raise ValueError("GI-HT and HT must use the same inference batching and checkpoint epoch.")


def plot_pe(out: Path, sweep: str, records: list[dict], scale: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"GI-HT": "#0072B2", "HT": "#D55E00", "Covariance": "#009E73", "Box-NNLS": "#CC79A7"}
    styles = {"GI-HT": "o-", "HT": "s--", "Covariance": "^-.", "Box-NNLS": "D:"}
    labels = {"L": "Pilot length L", "N": "Number of devices N", "SNR": "SNR (dB)"}
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for method in METHODS:
        rows = sorted((r for r in records if r["method"] == method), key=lambda r: r["value"])
        if not rows:
            continue
        values = np.array([r["min_pe"] for r in rows])
        if scale == "log":
            if np.any(values == 0):
                print(f"{method}: observed zero Pe stays zero in CSV; omitted on logarithmic axis.", flush=True)
            values = np.where(values > 0, values, np.nan)
        ax.plot([r["value"] for r in rows], values, styles[method], color=colors[method],
                label="Covariance (MLE)" if method == "Covariance" else method, markersize=4)
    xs = sorted({r["value"] for r in records})
    ax.set(xlabel=labels[sweep], ylabel=r"Minimum $P_e=P_M+P_F$")
    if scale == "log":
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.set_xticks(xs)
    if len(xs) > 1:
        ax.set_xlim(xs[0], xs[-1])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out / f"pe_vs_{sweep}.{suffix}", dpi=300)
    plt.close(fig)


def refresh_sweep(out: Path, sweep: str, field: str, points: list[tuple[float, str]], scale: str) -> list[dict]:
    records, point_curves = [], {}
    for value, label in points:
        curves = {}
        for method in METHODS:
            directory = out / "points" / label / method
            path = directory / "metrics.json"
            if not path.is_file():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["method"] != method or record["sweep"] != sweep or record["value"] != value:
                raise ValueError(f"Result identity disagrees with its directory: {path}")
            if record["system_config"][field] != value:
                raise ValueError(f"Saved scene disagrees with the sweep value: {path}")
            records.append(record)
            with (directory / "pm_pf_curve.csv").open(newline="", encoding="utf-8") as handle:
                curves[method] = [{k: (v if k == "method" else float(v)) for k, v in row.items()}
                                  for row in csv.DictReader(handle)]
        if curves:
            point_curves[label] = curves
    check_compatible(records, field)
    if not records:
        raise FileNotFoundError(f"No completed results to plot under {out}.")
    write_csv(out / f"pe_vs_{sweep}.csv", SUMMARY_FIELDS,
              [{key: record.get(key) for key in SUMMARY_FIELDS} for record in records])
    plot_pe(out, sweep, records, scale)
    for label, curves in point_curves.items():
        plot_comparison(out / "points" / label, curves)
    write_json(out / "summary_config.json", {"metric": METRIC, "sweep": sweep, "requested_values": [v for v, _ in points],
                                             "completed": [{"method": r["method"], "value": r["value"]} for r in records],
                                             "pe_axis_scale": scale})
    return records
