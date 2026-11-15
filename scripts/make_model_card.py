#!/usr/bin/env python
"""Generate docs/model_cards/<experiment>.md from a run_record.json (+ validation/test results if present).
Any quantity without a measured value is written as 'Not evaluated'."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    a = ap.parse_args()
    rec = json.loads((Path("outputs/checkpoints") / a.experiment / "run_record.json").read_text())
    m = rec["config"]["model"]
    lines = [
        f"# Model card — {a.experiment}",
        "",
        "**Research use only. Not a medical device.** See docs/RESEARCH_LIMITATIONS.md.",
        "",
        f"* Architecture: `{m['arch']}` (base channels {m.get('base_channels')}, depth {m.get('depth', 4)}, dropout {m.get('dropout', 0)}), random init, **no pretrained weights**",
        f"* Parameters: {rec['n_params']:,}",
        f"* Inputs: {rec['config']['data']['modalities']}, patch {rec['config']['data']['patch_size']}",
        f"* Training data: `{rec['config']['data']['dataset']}`, splits hash `{rec['splits_hash']}`, seed {rec['seed']}",
        f"* Device: {rec['device']}; epochs run {rec['epochs_run']} ({rec['stop_reason']})",
        f"* Selection metric: {rec['selection_metric']}; best = {rec['best_val_mean_dice'] if rec['best_val_mean_dice'] is not None else 'Not evaluated'}",
        f"* Checkpoint: `{rec['checkpoint_best']}`",
        "",
        "## Measured performance",
        "",
    ]
    for split in ("validation", "test"):
        p = Path("outputs/metrics") / a.experiment / f"{split}_summary.csv"
        if p.exists():
            import pandas as pd

            df = pd.read_csv(p)
            lines += [
                f"### {split} (patient-level bootstrap 95% CI)",
                "",
                "| region | Dice | HD95 (mm) |",
                "|---|---|---|",
            ]
            for r in ("WT", "TC", "ET"):
                d = df[(df.region == r) & (df.metric == "dice")].iloc[0]
                h = df[(df.region == r) & (df.metric == "hd95")].iloc[0]
                lines.append(
                    f"| {r} | {d['mean']:.3f} [{d.ci_lo:.3f}, {d.ci_hi:.3f}] | {h['mean']:.1f} [{h.ci_lo:.1f}, {h.ci_hi:.1f}] |"
                )
            lines.append("")
        else:
            lines += [f"### {split}", "", "Not evaluated", ""]
    lines += [
        "## Limitations",
        "",
        "Curated-challenge data only; no external or multi-site validation; uncertainty is predictive spread, not clinical risk.",
    ]
    out = Path("docs/model_cards")
    out.mkdir(exist_ok=True)
    (out / f"{a.experiment}.md").write_text("\n".join(lines))
    print("wrote", out / f"{a.experiment}.md")


if __name__ == "__main__":
    main()
