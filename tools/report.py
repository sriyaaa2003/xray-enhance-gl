"""Turn the study CSV into tables, figures and the results section of the README.

Settings are chosen on the tuning subjects only; every number reported is from the unseen subjects.
Intervals are bootstrap 95% intervals that resample whole subjects.

usage: python tools/report.py <study outputs> <inputs> <results folder>
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEVELS = [100000, 10000, 2000]
LEVEL_NAME = {100000: "low noise (100,000 photons)", 10000: "medium noise (10,000 photons)",
              2000: "high noise (2,000 photons)"}


def subject_interval(frame: pd.DataFrame, column: str, draws: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    per_subject = frame.groupby("id")[column].mean().to_numpy()
    rng = np.random.default_rng(seed)
    means = rng.choice(per_subject, size=(draws, len(per_subject)), replace=True).mean(axis=1)
    return float(per_subject.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_gain(data: pd.DataFrame, chosen: pd.DataFrame, baseline: pd.DataFrame, column: str):
    keys = ["id", "view", "photons"]
    joined = chosen.merge(baseline[keys + [column]], on=keys, suffixes=("", "_base"))
    joined["gain"] = joined[column] - joined[column + "_base"]
    return subject_interval(joined, "gain")


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def pick(data: pd.DataFrame, mask: pd.Series) -> str:
    """The setting with the best mean SSIM on the given tuning rows."""
    return data[mask].groupby("setting")["ssim"].mean().idxmax()


def denoise_section(data: pd.DataFrame) -> str:
    out = []
    for level in LEVELS:
        rows = []
        base = data[(data.photons == level) & (data.setting == "none|-") & (data.family == "denoise")]
        unseen_base = base[base.split == "unseen"]
        for method in ("none", "gaussian", "bilateral"):
            tuning = (data.photons == level) & (data.family == "denoise") & (data.method == method) & (data.split == "tuning")
            setting = pick(data, tuning)
            chosen = data[(data.photons == level) & (data.family == "denoise") & (data.setting == setting)
                          & (data.split == "unseen")]
            gain, low, high = paired_gain(data, chosen, unseen_base, "ssim")
            rows.append([method, setting.split("|")[1], f"{chosen.psnr.mean():.2f}", f"{chosen.ssim.mean():.3f}",
                         "-" if method == "none" else f"{gain:+.3f} ({low:+.3f} to {high:+.3f})",
                         f"{chosen.remaining_noise.mean():.4f}", f"{chosen.edge_retention.mean():.2f}"])
        out.append(f"**{LEVEL_NAME[level]}**\n\n" + table(
            ["filter", "setting chosen on tuning subjects", "PSNR (dB)", "SSIM", "SSIM change vs none (95% interval)",
             "noise left", "edge strength kept"], rows))
    return "\n\n".join(out)


def sharpen_section(data: pd.DataFrame) -> str:
    rows = []
    unseen = data[(data.photons == 10000) & (data.split == "unseen")]
    for setting in ["none|-"] + sorted(unseen[unseen.family == "sharpen"].setting.unique()):
        chosen = unseen[(unseen.setting == setting) & (unseen.family.isin(["sharpen", "denoise"]))]
        rows.append([setting.replace("|", ", "), f"{chosen.remaining_noise.mean():.4f}",
                     f"{chosen.edge_retention.mean():.2f}", f"{chosen.ssim.mean():.3f}"])
    return table(["method (medium noise)", "noise left", "edge strength kept", "SSIM"], rows)


def contrast_section(data: pd.DataFrame) -> str:
    rows = []
    frame = data[(data.photons == 10000) & (data.split == "unseen") & (data.family == "contrast")]
    order = list(dict.fromkeys(frame.setting))
    base = frame[frame.setting == "none|-"]
    for setting in order:
        chosen = frame[frame.setting == setting]
        step = chosen.bone_contrast.mean() / base.bone_contrast.mean()
        cnr = chosen.bone_cnr.mean() / base.bone_cnr.mean()
        rows.append([setting.replace("|", ", "), f"{chosen.bone_contrast.mean():.4f}", f"{step:.2f}x",
                     f"{chosen.bone_cnr.mean():.3f}", f"{cnr:.2f}x", f"{100 * chosen.clipped_fraction.mean():.2f}%"])
    return table(["method (medium noise, unseen subjects)", "bone-edge step on screen", "vs raw",
                  "bone-edge step / local spread", "vs raw", "pixels clipped to black or white"], rows)


def transfer_section(data: pd.DataFrame) -> str:
    """Does a setting tuned on one kind of image work on another?"""
    bilateral = (data.family == "denoise") & (data.method == "bilateral")

    def score(setting: str, mask: pd.Series) -> float:
        return float(data[mask & bilateral & (data.setting == setting) & (data.split == "unseen")].ssim.mean())

    region_rows = []
    for level in LEVELS:
        picks = {r: pick(data, bilateral & (data.split == "tuning") & (data.region == r) & (data.photons == level))
                 for r in ("thorax", "pelvis")}
        for source, target in (("thorax", "pelvis"), ("pelvis", "thorax")):
            where = (data.region == target) & (data.photons == level)
            own, borrowed = score(picks[target], where), score(picks[source], where)
            region_rows.append([LEVEL_NAME[level], f"{source} to {target}", picks[source].split("|")[1],
                                picks[target].split("|")[1], f"{borrowed - own:+.4f}"])
    dose_rows = []
    picks = {l: pick(data, bilateral & (data.split == "tuning") & (data.photons == l)) for l in LEVELS}
    for source in LEVELS:
        row = [f"{LEVEL_NAME[source]}: {picks[source].split('|')[1]}"]
        for target in LEVELS:
            where = data.photons == target
            row.append(f"{score(picks[source], where):.3f}")
        dose_rows.append(row)
    return ("**Across body regions** (bilateral filter; SSIM on unseen subjects with the borrowed setting minus "
            "SSIM with the region's own setting)\n\n"
            + table(["noise", "tuned on to used on", "borrowed setting", "own setting", "SSIM change"], region_rows)
            + "\n\n**Across noise levels** (rows: where the setting was tuned; cells: SSIM on unseen subjects)\n\n"
            + table(["tuned at"] + [LEVEL_NAME[l] for l in LEVELS], dose_rows))


def tradeoff_figure(data: pd.DataFrame, path: Path) -> None:
    unseen = data[(data.photons == 10000) & (data.split == "unseen") & data.family.isin(["denoise", "sharpen"])]
    means = unseen.groupby(["method"]+["setting"])[["remaining_noise", "edge_retention"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for method, group in means.groupby("method"):
        ax.scatter(group.remaining_noise, group.edge_retention, label=method, s=36)
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("noise left (lower is better)")
    ax.set_ylabel("edge strength kept (1 = same as the noise-free image)")
    ax.set_title("Smoothing and sharpening trade-off, medium noise, unseen subjects")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def read_pfm(path: Path) -> np.ndarray:
    with open(path, "rb") as handle:
        assert handle.readline().strip() == b"Pf"
        width, height = map(int, handle.readline().split())
        handle.readline()
        return np.flipud(np.frombuffer(handle.read(), dtype="<f4").reshape(height, width))


def example_figure(outputs: Path, path: Path) -> None:
    panels = [("raw, medium noise", "example_noisy.pfm"),
              ("automatic window, 2-98%", "example_auto-window_percent=2.pfm"),
              ("histogram equalisation", "example_equalise_-.pfm"),
              ("CLAHE, clip 2", "example_clahe_clip=2.pfm"),
              ("bilateral then CLAHE", "example_bilateral+clahe_clip=2.pfm"),
              ("local contrast, gain 2", "example_local-contrast_gain=2.pfm")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.4))
    for ax, (title, name) in zip(axes.ravel(), panels):
        ax.imshow(read_pfm(outputs / name), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(outputs: str, inputs: str, results: str) -> None:
    outputs_path, inputs_path, results_path = Path(outputs), Path(inputs), Path(results)
    results_path.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(outputs_path / "results.csv", dtype={"id": str, "params": str})
    data["setting"] = data.method + "|" + data.params
    cases = pd.read_csv(inputs_path / "cases.csv", dtype={"id": str})
    counts = cases.groupby(["region", "split"]).agg(subjects=("id", "nunique"), images=("id", "size")).reset_index()

    tradeoff_figure(data, results_path / "tradeoff.png")
    example_figure(outputs_path, results_path / "examples.png")
    for name in ("results.csv", "run_info.txt"):
        shutil.copy(outputs_path / name, results_path / name)
    shutil.copy(inputs_path / "cases.csv", results_path / "cases.csv")

    summary = "\n\n".join([
        "## Results",
        f"Generated by `tools/report.py` from `results/results.csv` ({len(data):,} rows). "
        f"{cases.id.nunique()} subjects, {len(cases)} projections.",
        table(["region", "split", "subjects", "projections"],
              [[r.region, r.split, str(r.subjects), str(r.images)] for r in counts.itertuples()]),
        (results_path / "run_info.txt").read_text().strip().replace("\n", "  \n"),
        "### Noise reduction", denoise_section(data),
        "### Sharpening brings noise back", sharpen_section(data), "![trade-off](results/tradeoff.png)",
        "### Contrast methods", contrast_section(data), "![examples](results/examples.png)",
        "### Do settings carry over?", transfer_section(data),
    ]) + "\n"
    (results_path / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main(*sys.argv[1:4])
