"""
VIGIL Phase 1 — Dataset Inspection Script
MVTec AD 2 Vial Dataset Analysis

This script inspects the actual dataset files to produce:
- Directory structure map
- Per-split image counts and class distribution
- Image resolution and channel statistics
- File size distribution
- Naming convention analysis (regular, overexposed, underexposed, shift variants)
- Visualizations of normal and anomalous samples
- Ground-truth mask overlay visualizations
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_ROOT = Path(r"E:\CODING\VIGIL\dataset")
REPORT_DIR = Path(r"E:\CODING\VIGIL\reports\dataset_inspection")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ── Utility functions ──────────────────────────────────────────────────────────
def parse_filename(fname: str) -> dict:
    """Parse MVTec AD 2 naming convention: NNN_variant[_mask].png"""
    stem = Path(fname).stem
    parts = stem.split("_")
    info = {"filename": fname, "index": parts[0]}

    if stem.endswith("_mask"):
        info["is_mask"] = True
        variant_parts = parts[1:-1]
    else:
        info["is_mask"] = False
        variant_parts = parts[1:]

    info["variant"] = "_".join(variant_parts) if variant_parts else "unknown"
    return info


def get_image_info(filepath: Path) -> dict:
    """Extract image metadata without loading full pixel data."""
    try:
        with Image.open(filepath) as img:
            return {
                "path": str(filepath),
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "channels": len(img.getbands()),
                "format": img.format,
                "size_bytes": filepath.stat().st_size,
            }
    except Exception as e:
        return {"path": str(filepath), "error": str(e)}


def discover_images(directory: Path, extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
    """Recursively find all image files."""
    images = []
    if not directory.exists():
        return images
    for ext in extensions:
        images.extend(sorted(directory.rglob(f"*{ext}")))
    return sorted(set(images))


# ── 1. Directory structure ─────────────────────────────────────────────────────
print("=" * 80)
print("VIGIL — MVTec AD 2 Vial Dataset Inspection")
print("=" * 80)

print("\n1. DIRECTORY STRUCTURE")
print("-" * 40)

structure = {}
for split_dir in sorted(DATASET_ROOT.iterdir()):
    if not split_dir.is_dir():
        continue
    split_name = split_dir.name
    structure[split_name] = {}
    for sub in sorted(split_dir.iterdir()):
        if sub.is_dir():
            # Count files in subdirectory (recursive)
            files = list(sub.rglob("*.png"))
            structure[split_name][sub.name] = len(files)
            # Check for deeper nesting
            for subsub in sorted(sub.iterdir()):
                if subsub.is_dir():
                    subfiles = list(subsub.rglob("*.png"))
                    structure[split_name][f"{sub.name}/{subsub.name}"] = len(subfiles)
        else:
            structure[split_name]["_files"] = structure[split_name].get("_files", 0) + 1

for split, contents in structure.items():
    print(f"\n  {split}/")
    for name, count in contents.items():
        print(f"    {name}: {count} images")

# ── 2. Per-split detailed analysis ────────────────────────────────────────────
print("\n\n2. PER-SPLIT IMAGE ANALYSIS")
print("-" * 40)

all_stats = {}

splits = {
    "train/good": DATASET_ROOT / "train" / "good",
    "validation/good": DATASET_ROOT / "validation" / "good",
    "test_public/good": DATASET_ROOT / "test_public" / "good",
    "test_public/bad": DATASET_ROOT / "test_public" / "bad",
    "test_public/ground_truth/bad": DATASET_ROOT / "test_public" / "ground_truth" / "bad",
    "test_private": DATASET_ROOT / "test_private",
    "test_private_mixed": DATASET_ROOT / "test_private_mixed",
}

for split_name, split_path in splits.items():
    images = discover_images(split_path)
    if not images:
        print(f"\n  {split_name}: NO IMAGES FOUND")
        continue

    # Gather info for first/last and a sample
    resolutions = Counter()
    modes = Counter()
    variants = Counter()
    sizes = []

    for img_path in images:
        info = get_image_info(img_path)
        if "error" in info:
            continue
        res = f"{info['width']}x{info['height']}"
        resolutions[res] += 1
        modes[info["mode"]] += 1
        sizes.append(info["size_bytes"])

        parsed = parse_filename(img_path.name)
        variants[parsed["variant"]] += 1

    all_stats[split_name] = {
        "count": len(images),
        "resolutions": dict(resolutions),
        "modes": dict(modes),
        "variants": dict(variants),
        "size_min_kb": min(sizes) / 1024 if sizes else 0,
        "size_max_kb": max(sizes) / 1024 if sizes else 0,
        "size_mean_kb": np.mean(sizes) / 1024 if sizes else 0,
        "sample_paths": [str(images[0]), str(images[len(images)//2]), str(images[-1])],
    }

    print(f"\n  {split_name}:")
    print(f"    Count: {len(images)}")
    print(f"    Resolutions: {dict(resolutions)}")
    print(f"    Color modes: {dict(modes)}")
    print(f"    Variants: {dict(variants)}")
    print(f"    Size range: {min(sizes)/1024:.1f} KB — {max(sizes)/1024:.1f} KB (mean {np.mean(sizes)/1024:.1f} KB)")

# ── 3. Pixel-level statistics on a sample ─────────────────────────────────────
print("\n\n3. PIXEL-LEVEL STATISTICS (sampled)")
print("-" * 40)

sample_dirs = {
    "train/good": DATASET_ROOT / "train" / "good",
    "test_public/good": DATASET_ROOT / "test_public" / "good",
    "test_public/bad": DATASET_ROOT / "test_public" / "bad",
}

for name, path in sample_dirs.items():
    images = discover_images(path)[:10]  # First 10 for speed
    if not images:
        continue
    pixel_means = []
    pixel_stds = []
    for img_path in images:
        arr = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        pixel_means.append(arr.mean(axis=(0, 1)))
        pixel_stds.append(arr.std(axis=(0, 1)))

    mean_rgb = np.mean(pixel_means, axis=0)
    std_rgb = np.mean(pixel_stds, axis=0)
    print(f"\n  {name} (n={len(images)} sampled):")
    print(f"    Mean RGB: [{mean_rgb[0]:.4f}, {mean_rgb[1]:.4f}, {mean_rgb[2]:.4f}]")
    print(f"    Std  RGB: [{std_rgb[0]:.4f}, {std_rgb[1]:.4f}, {std_rgb[2]:.4f}]")

# ── 4. Naming convention & variant analysis ───────────────────────────────────
print("\n\n4. NAMING CONVENTION & DOMAIN-SHIFT VARIANTS")
print("-" * 40)

bad_images = discover_images(DATASET_ROOT / "test_public" / "bad")
bad_parsed = [parse_filename(f.name) for f in bad_images]
bad_indices = sorted(set(p["index"] for p in bad_parsed))
bad_variants = Counter(p["variant"] for p in bad_parsed)

print(f"  Unique defective vials (by index): {len(bad_indices)}")
print(f"  Unique indices: {bad_indices}")
print(f"  Variants per defect: {dict(bad_variants)}")
print(f"  Total bad images: {len(bad_images)}")
print(f"  Images per defective vial: {len(bad_images) / len(bad_indices):.1f}" if bad_indices else "  N/A")

# Check ground truth masks alignment
gt_masks = discover_images(DATASET_ROOT / "test_public" / "ground_truth" / "bad")
gt_parsed = [parse_filename(f.name) for f in gt_masks]
gt_indices = sorted(set(p["index"] for p in gt_parsed))

print(f"\n  Ground-truth masks:")
print(f"    Total mask files: {len(gt_masks)}")
print(f"    Unique defect indices: {len(gt_indices)}")
print(f"    Indices match bad images: {gt_indices == bad_indices}")

# ── 5. test_private and test_private_mixed analysis ───────────────────────────
print("\n\n5. TEST_PRIVATE AND TEST_PRIVATE_MIXED ANALYSIS")
print("-" * 40)

tp_images = discover_images(DATASET_ROOT / "test_private")
tp_parsed = [parse_filename(f.name) for f in tp_images]
tp_variants = Counter(p["variant"] for p in tp_parsed)
print(f"  test_private: {len(tp_images)} images")
print(f"    All named '*_regular.png': {'regular' in tp_variants and len(tp_variants) == 1}")
print(f"    Variants: {dict(tp_variants)}")

tpm_images = discover_images(DATASET_ROOT / "test_private_mixed")
tpm_parsed = [parse_filename(f.name) for f in tpm_images]
tpm_variants = Counter(p["variant"] for p in tpm_parsed)
tpm_sizes = [f.stat().st_size for f in tpm_images]
print(f"\n  test_private_mixed: {len(tpm_images)} images")
print(f"    Variants: {dict(tpm_variants)}")

# Size analysis for mixed — helps distinguish normal from anomalous
sizes_regular_range = [s for p, s in zip(tp_parsed, [f.stat().st_size for f in tp_images])]
mixed_small = sum(1 for s in tpm_sizes if s < 500000)
mixed_large = sum(1 for s in tpm_sizes if s >= 500000 and s < 1000000)
mixed_xlarge = sum(1 for s in tpm_sizes if s >= 1000000)
print(f"    Size distribution: <500KB: {mixed_small}, 500KB-1MB: {mixed_large}, >1MB: {mixed_xlarge}")


# ── 6. Save statistics JSON ──────────────────────────────────────────────────
stats_path = REPORT_DIR / "dataset_statistics.json"
with open(stats_path, "w") as f:
    json.dump(all_stats, f, indent=2, default=str)
print(f"\n  Statistics saved to: {stats_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n6. GENERATING VISUALIZATIONS")
print("-" * 40)

# ── 6a. Representative normal samples (train) ────────────────────────────────
train_good = sorted((DATASET_ROOT / "train" / "good").glob("*.png"))
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Representative NORMAL Samples (train/good)", fontsize=16, fontweight="bold")
for i, ax in enumerate(axes.flat):
    idx = i * (len(train_good) // 8)
    img = Image.open(train_good[idx])
    ax.imshow(img)
    ax.set_title(train_good[idx].name, fontsize=9)
    ax.axis("off")
plt.tight_layout()
fig.savefig(REPORT_DIR / "normal_samples_train.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] normal_samples_train.png")

# ── 6b. Representative anomalous (bad) samples ──────────────────────────────
bad_regular = sorted([f for f in bad_images if "_regular" in f.name])
fig, axes = plt.subplots(3, 5, figsize=(25, 15))
fig.suptitle("Anomalous Samples — Regular Views (test_public/bad)", fontsize=16, fontweight="bold")
for i, ax in enumerate(axes.flat):
    if i < len(bad_regular):
        img = Image.open(bad_regular[i])
        ax.imshow(img)
        ax.set_title(bad_regular[i].name, fontsize=9)
    ax.axis("off")
plt.tight_layout()
fig.savefig(REPORT_DIR / "anomalous_samples_regular.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] anomalous_samples_regular.png")

# ── 6c. Domain-shift variants for a single defect ───────────────────────────
defect_idx = "000"
defect_variants = sorted([f for f in bad_images if f.name.startswith(f"{defect_idx}_")])
mask_variants = sorted([f for f in gt_masks if f.name.startswith(f"{defect_idx}_")])

fig, axes = plt.subplots(2, len(defect_variants), figsize=(4*len(defect_variants), 8))
fig.suptitle(f"Domain-Shift Variants for Defect #{defect_idx}", fontsize=16, fontweight="bold")
for i, (img_f, mask_f) in enumerate(zip(defect_variants, mask_variants)):
    img = np.array(Image.open(img_f))
    mask = np.array(Image.open(mask_f))
    axes[0, i].imshow(img)
    axes[0, i].set_title(img_f.stem.replace(f"{defect_idx}_", ""), fontsize=9)
    axes[0, i].axis("off")
    axes[1, i].imshow(mask, cmap="hot")
    axes[1, i].set_title("Mask", fontsize=9)
    axes[1, i].axis("off")
plt.tight_layout()
fig.savefig(REPORT_DIR / "domain_shift_variants.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] domain_shift_variants.png")

# ── 6d. Ground-truth mask overlay ────────────────────────────────────────────
fig, axes = plt.subplots(4, 3, figsize=(15, 20))
fig.suptitle("Ground-Truth Anomaly Mask Overlays (test_public/bad)", fontsize=16, fontweight="bold")
col_labels = ["Original Image", "Anomaly Mask", "Overlay"]

for row_idx in range(4):
    if row_idx >= len(bad_regular):
        break
    img_path = bad_regular[row_idx]
    # Find corresponding mask
    mask_name = img_path.stem + "_mask.png"
    mask_path = DATASET_ROOT / "test_public" / "ground_truth" / "bad" / mask_name

    img = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))

    # Normalize mask to binary if needed
    if mask.max() > 1:
        mask_binary = (mask > 127).astype(np.float32)
    else:
        mask_binary = mask.astype(np.float32)

    # Handle multi-channel mask
    if mask_binary.ndim == 3:
        mask_binary = mask_binary[:, :, 0]

    # Create overlay (image is now RGB)
    overlay = img.copy().astype(np.float32)
    # Red channel overlay where mask is active
    red_overlay = np.zeros_like(overlay)
    red_overlay[:, :, 0] = 255
    alpha = 0.4
    mask_3d = np.stack([mask_binary] * 3, axis=-1)
    overlay = overlay * (1 - mask_3d * alpha) + red_overlay * mask_3d * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    axes[row_idx, 0].imshow(img)
    axes[row_idx, 0].set_title(f"Original: {img_path.name}", fontsize=9)
    axes[row_idx, 0].axis("off")

    axes[row_idx, 1].imshow(mask, cmap="hot")
    axes[row_idx, 1].set_title(f"Mask: {mask_name}", fontsize=9)
    axes[row_idx, 1].axis("off")

    axes[row_idx, 2].imshow(overlay)
    axes[row_idx, 2].set_title("Overlay (mask in red)", fontsize=9)
    axes[row_idx, 2].axis("off")

plt.tight_layout()
fig.savefig(REPORT_DIR / "mask_overlays.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] mask_overlays.png")


# ── 6e. Resolution & file size distributions ─────────────────────────────────
# Collect all resolutions and sizes across splits
all_resolutions = Counter()
all_file_sizes = []
split_sizes = defaultdict(list)

for split_name, split_path in splits.items():
    if "ground_truth" in split_name:
        continue
    images = discover_images(split_path)
    for img_path in images:
        info = get_image_info(img_path)
        if "error" not in info:
            all_resolutions[f"{info['width']}x{info['height']}"] += 1
            all_file_sizes.append(info["size_bytes"] / 1024)
            split_sizes[split_name].append(info["size_bytes"] / 1024)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Dataset Distribution Analysis", fontsize=16, fontweight="bold")

# Resolution bar chart
res_labels = list(all_resolutions.keys())
res_counts = list(all_resolutions.values())
ax1.barh(res_labels, res_counts, color="steelblue")
ax1.set_xlabel("Count")
ax1.set_title("Image Resolution Distribution")
for i, v in enumerate(res_counts):
    ax1.text(v + 1, i, str(v), va="center", fontsize=9)

# File size histogram per split
colors = plt.cm.tab10.colors
for i, (sname, sizes) in enumerate(split_sizes.items()):
    ax2.hist(sizes, bins=30, alpha=0.5, label=sname, color=colors[i % len(colors)])
ax2.set_xlabel("File Size (KB)")
ax2.set_ylabel("Count")
ax2.set_title("File Size Distribution by Split")
ax2.legend(fontsize=8)

plt.tight_layout()
fig.savefig(REPORT_DIR / "distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] distributions.png")

# ── 6f. Normal vs anomalous comparison ───────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
fig.suptitle("Normal (top) vs Anomalous (bottom) Comparison", fontsize=16, fontweight="bold")

# Top row: normal samples
for i in range(5):
    idx = i * (len(train_good) // 5)
    img = Image.open(train_good[idx])
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"NORMAL: {train_good[idx].name}", fontsize=9, color="green")
    axes[0, i].axis("off")

# Bottom row: anomalous samples
for i in range(5):
    if i < len(bad_regular):
        img = Image.open(bad_regular[i])
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"ANOMALOUS: {bad_regular[i].name}", fontsize=9, color="red")
    axes[1, i].axis("off")

plt.tight_layout()
fig.savefig(REPORT_DIR / "normal_vs_anomalous.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] normal_vs_anomalous.png")

# ── Check mask resolution vs image resolution ───────────────────────────────
print("\n\n7. MASK vs IMAGE RESOLUTION CHECK")
print("-" * 40)
sample_img = Image.open(bad_regular[0])
sample_mask_name = bad_regular[0].stem + "_mask.png"
sample_mask = Image.open(DATASET_ROOT / "test_public" / "ground_truth" / "bad" / sample_mask_name)
print(f"  Sample image resolution: {sample_img.size}, mode: {sample_img.mode}")
print(f"  Sample mask resolution: {sample_mask.size}, mode: {sample_mask.mode}")
print(f"  Resolutions match: {sample_img.size == sample_mask.size}")

# Check mask unique values
mask_arr = np.array(sample_mask)
print(f"  Mask shape: {mask_arr.shape}")
print(f"  Mask unique values: {np.unique(mask_arr)}")
print(f"  Mask dtype: {mask_arr.dtype}")

# Check multiple masks
print("\n  Checking mask properties across all defects:")
for i in range(min(5, len(bad_regular))):
    mn = bad_regular[i].stem + "_mask.png"
    mp = DATASET_ROOT / "test_public" / "ground_truth" / "bad" / mn
    m = np.array(Image.open(mp))
    defect_pixels = np.sum(m > 127)
    total_pixels = m.size if m.ndim == 2 else m[:,:,0].size
    print(f"    Defect {i}: shape={m.shape}, unique={np.unique(m).tolist()}, "
          f"defect_area={defect_pixels/total_pixels*100:.2f}%")


print("\n\n" + "=" * 80)
print("DATASET INSPECTION COMPLETE")
print("=" * 80)
print(f"\nAll reports saved to: {REPORT_DIR}")
