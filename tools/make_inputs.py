"""Turn CT scans from the TotalSegmentator small subset into 2D X-ray-like projections.

For each chosen subject the CT is converted from Hounsfield units to attenuation and summed along one
axis (a parallel-beam projection, not a real X-ray geometry). The expert-corrected bone segmentations
are projected the same way and mark where bone outlines are.

usage: python tools/make_inputs.py <dataset.zip> <output folder>
"""
from __future__ import annotations

import csv
import gzip
import re
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

MU_WATER = 0.02          # attenuation of water per mm, about 60-70 keV
DISPLAY_SCALE = 10.0     # line integral shown as display value 1 (the same for every image)
PER_REGION = 12          # subjects per body region
MIN_SIDE = 128           # pixels
BONE_THICKNESS_MM = 6.0  # bone path length along the ray that counts as "bone here"
BONE_PATTERN = re.compile(r"(vertebrae_.+|rib_.+|hip_.+|sacrum|femur_.+|scapula_.+|clavicula_.+|humerus_.+|sternum|skull)\.nii\.gz$")
LUNGS = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
         "lung_middle_lobe_right", "lung_lower_lobe_right"]


def load(archive: zipfile.ZipFile, member: str) -> nib.Nifti1Image:
    image = nib.Nifti1Image.from_bytes(gzip.decompress(archive.read(member)))
    return nib.as_closest_canonical(image)  # axes become right, anterior, superior


def voxel_count(archive, names: set[str], member: str) -> int:
    return int(np.count_nonzero(np.asanyarray(load(archive, member).dataobj))) if member in names else 0


def region_of(archive, names: set[str], folder: str) -> str | None:
    seg = folder + "segmentations/"
    hips = min(voxel_count(archive, names, seg + f"hip_{side}.nii.gz") for side in ("left", "right"))
    lungs = sum(voxel_count(archive, names, seg + f"{name}.nii.gz") for name in LUNGS)
    is_pelvis, is_thorax = hips > 20_000, lungs > 200_000
    if is_pelvis == is_thorax:      # neither, or a long scan that contains both: skip, to keep regions apart
        return None
    return "pelvis" if is_pelvis else "thorax"


def write_pfm(path: Path, image: np.ndarray) -> None:
    height, width = image.shape
    with open(path, "wb") as out:
        out.write(f"Pf\n{width} {height}\n-1.0\n".encode())
        out.write(np.flipud(image).astype("<f4").tobytes())


def project(volume: np.ndarray, view: str, spacing: tuple[float, float, float]) -> np.ndarray:
    """Sum a right/anterior/superior volume into an image whose top row is the most superior."""
    if view == "ap":   # along anterior-posterior; patient right on the image left
        summed = volume.sum(axis=1) * spacing[1]
        return summed[::-1, ::-1].T
    summed = volume.sum(axis=0) * spacing[0]  # lateral; anterior on the image left
    return summed[::-1, ::-1].T


def crop_box(body: np.ndarray, margin: int = 6) -> tuple[slice, slice]:
    rows, cols = np.nonzero(body)
    return (slice(max(rows.min() - margin, 0), rows.max() + margin + 1),
            slice(max(cols.min() - margin, 0), cols.max() + margin + 1))


def main(zip_path: str, out_folder: str) -> None:
    out = Path(out_folder)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        folders = sorted(n[: -len("ct.nii.gz")] for n in names if n.endswith("/ct.nii.gz"))
        chosen: dict[str, list[str]] = {"thorax": [], "pelvis": []}
        for folder in folders:
            if all(len(v) >= PER_REGION for v in chosen.values()):
                break
            region = region_of(archive, names, folder)
            if region and len(chosen[region]) < PER_REGION:
                chosen[region].append(folder)
                print(f"{folder} -> {region}", flush=True)

        rows, clipped_shares = [], []
        for region, subjects in chosen.items():
            for position, folder in enumerate(subjects):
                subject = folder.rstrip("/").split("/")[-1]
                split = "tuning" if position % 3 == 0 else "unseen"  # split by subject, so views stay together
                ct = load(archive, folder + "ct.nii.gz")
                spacing = tuple(float(z) for z in ct.header.get_zooms()[:3])
                hu = np.asarray(ct.dataobj, dtype=np.float32)
                mu = MU_WATER * np.clip(1.0 + hu / 1000.0, 0.0, None)
                bone = np.zeros(hu.shape, dtype=bool)
                for member in sorted(names):
                    if member.startswith(folder + "segmentations/") and BONE_PATTERN.search(member):
                        bone |= np.asanyarray(load(archive, member).dataobj) > 0
                for view in ("ap", "lateral"):
                    line = project(mu, view, spacing)
                    bone_mm = project(bone.astype(np.float32), view, spacing)
                    body = line > 0.5
                    if not body.any():
                        continue
                    box = crop_box(body)
                    line, bone_mm, body = line[box], bone_mm[box], body[box]
                    if min(line.shape) < MIN_SIDE or not (bone_mm >= BONE_THICKNESS_MM).any():
                        print(f"skipped {subject} {view}: image too small or no bone", flush=True)
                        continue
                    clipped_shares.append(float((line > DISPLAY_SCALE).mean()))
                    stem = f"{subject}_{view}"
                    write_pfm(out / f"{stem}_clean.pfm", np.clip(line / DISPLAY_SCALE, 0.0, 1.0))
                    write_pfm(out / f"{stem}_bone.pfm", (bone_mm >= BONE_THICKNESS_MM).astype(np.float32))
                    write_pfm(out / f"{stem}_body.pfm", body.astype(np.float32))
                    rows.append([subject, view, region, split, DISPLAY_SCALE,
                                 f"{stem}_clean.pfm", f"{stem}_bone.pfm", f"{stem}_body.pfm"])

    with open(out / "cases.csv", "w", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "view", "region", "split", "scale", "image", "bone", "body"])
        writer.writerows(rows)
    example = next(r for r in rows if r[2] == "pelvis" and r[3] == "unseen" and r[1] == "ap")
    (out / "example.txt").write_text(f"{example[0]}_{example[1]}")
    print(f"{len(rows)} projections; largest share of pixels above the display scale: {max(clipped_shares):.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
