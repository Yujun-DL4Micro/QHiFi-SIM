import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from natsort import natsorted
from skimage import filters, measure, morphology, restoration
from skimage.morphology import disk, reconstruction


def natural_sort(files):
    """Return files in natural sort order."""
    return natsorted(files)


def list_tif(folder: Path):
    """List TIFF files in a folder."""
    return [
        p.name
        for p in folder.iterdir()
        if p.suffix.lower() in (".tif", ".tiff")
    ]


def normalize_image(img):
    """Normalize an image to the range [0, 1]."""
    img = img.astype(np.float32)
    max_val = float(img.max())
    if max_val <= 0:
        return img
    return img / max_val


def segment_lipid(
    img,
    gauss_sigma=1,
    sauvola_w=25,
    sauvola_k=0.15,
    area_lim=(40, 800),
    circ_lim=0.75,
):
    """Segment lipid droplets and return a labeled mask."""
    img_f = normalize_image(img)

    img_d = filters.gaussian(img_f, sigma=gauss_sigma, preserve_range=False)
    img_bg = restoration.rolling_ball(img_d, radius=50)
    img_d = img_d - img_bg

    thresh = filters.threshold_sauvola(
        img_d, window_size=sauvola_w, k=sauvola_k
    )
    bw = (img_d > thresh).astype(np.uint8)

    bw = morphology.opening(bw, footprint=disk(2))
    seed = morphology.erosion(bw, footprint=disk(2))
    bw = reconstruction(seed, bw, method="dilation")

    label = measure.label(bw, connectivity=2)
    props = measure.regionprops(label)

    for p in props:
        if (
            p.area < area_lim[0]
            or p.area > area_lim[1]
            or p.eccentricity > np.sqrt(1 - circ_lim ** 2)
        ):
            label[label == p.label] = 0

    label = morphology.label(label, connectivity=2)
    return label


def to_bgr_for_marking(img):
    """Convert an image to an 8-bit BGR image for annotation."""
    img = normalize_image(img)
    img8 = (img * 255).astype(np.uint8)

    if img8.ndim == 2:
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

    return img8


def draw_boxes(img_bgr, label, thickness=3):
    """Draw gold bounding boxes and index numbers."""
    img_vis = img_bgr.copy()
    props = measure.regionprops(label)
    gold_color = (0, 215, 255)

    for idx, p in enumerate(props, 1):
        minr, minc, maxr, maxc = p.bbox
        cv2.rectangle(
            img_vis,
            (minc, minr),
            (maxc, maxr),
            gold_color,
            thickness,
        )
        cv2.putText(
            img_vis,
            str(idx),
            (minc, minr - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            gold_color,
            2,
        )

    return img_vis


def linear_regression(x, y):
    """Fit y = slope * x + intercept and return slope, intercept, and R2."""
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        r2 = np.nan
    else:
        ss_res = np.sum((y - (slope * x + intercept)) ** 2)
        r2 = 1 - ss_res / ss_tot

    return slope, intercept, r2


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Quantify lipid droplet intensities in WF, SIM, and QSIM TIFF images."
        )
    )

    parser.add_argument(
        "--wf-folder",
        type=Path,
        required=True,
        help="Folder containing 10 WF TIFF images.",
    )
    parser.add_argument(
        "--sim-folders",
        type=Path,
        nargs="+",
        required=True,
        help="Folders containing SIM TIFF images.",
    )
    parser.add_argument(
        "--qsim-folders",
        type=Path,
        nargs="+",
        required=True,
        help="Folders containing QSIM TIFF images in the same algorithm order.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root folder.",
    )

    return parser.parse_args()


def main(args):
    """Run lipid droplet quantification."""
    args.out_root.mkdir(parents=True, exist_ok=True)

    if len(args.sim_folders) != len(args.qsim_folders):
        raise ValueError(
            "The number of SIM and QSIM folders must be the same."
        )

    wf_files = natural_sort(list_tif(args.wf_folder))
    if len(wf_files) != 10:
        raise ValueError(
            f"WF folder must contain 10 TIFF images, found {len(wf_files)}."
        )

    masks_wf = {}

    for i, wf_file in enumerate(wf_files, 1):
        print(f"[Segment] WF {i}/10: {wf_file}")
        wf_img = tifffile.imread(str(args.wf_folder / wf_file))
        masks_wf[wf_file] = segment_lipid(wf_img)

    for sim_path, qsim_path in zip(args.sim_folders, args.qsim_folders):
        algo_name = qsim_path.parent.name
        print(f"\n===== Processing pair: {algo_name} =====")

        out_sub = args.out_root / algo_name
        (out_sub / "marked").mkdir(parents=True, exist_ok=True)
        (out_sub / "excel").mkdir(parents=True, exist_ok=True)
        (out_sub / "linreg").mkdir(parents=True, exist_ok=True)

        sim_files = natural_sort(list_tif(sim_path))
        qsim_files = natural_sort(list_tif(qsim_path))

        if len(sim_files) != 10:
            raise ValueError(
                f"{algo_name}: SIM folder must contain 10 TIFF images, "
                f"found {len(sim_files)}."
            )

        if len(qsim_files) != 10:
            raise ValueError(
                f"{algo_name}: QSIM folder must contain 10 TIFF images, "
                f"found {len(qsim_files)}."
            )

        for i in range(10):
            print(f"  Processing image {i + 1}/10: {wf_files[i]}")

            img_sim = tifffile.imread(str(sim_path / sim_files[i]))
            img_qsim = tifffile.imread(str(qsim_path / qsim_files[i]))
            img_wf = tifffile.imread(str(args.wf_folder / wf_files[i]))

            for im in (img_sim, img_qsim, img_wf):
                if im is None:
                    raise ValueError(f"Failed to read image: {wf_files[i]}")

            label = masks_wf[wf_files[i]]

            props = measure.regionprops(label, intensity_image=img_wf)
            data = []

            for idx, p in enumerate(props, 1):
                coords = p.coords

                val_sim = img_sim[coords[:, 0], coords[:, 1]].mean()
                val_qsim = img_qsim[coords[:, 0], coords[:, 1]].mean()
                val_wf = img_wf[coords[:, 0], coords[:, 1]].mean()

                data.append([idx, val_sim, val_qsim, val_wf])

            df = pd.DataFrame(
                data,
                columns=["ID", "mean_SIM", "mean_QSIM", "mean_WF"],
            )
            df.to_excel(
                out_sub / "excel" / f"{Path(wf_files[i]).stem}.xlsx",
                index=False,
            )

            sim_bgr = to_bgr_for_marking(img_sim)
            marked = draw_boxes(sim_bgr, label)
            marked_rgb = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)

            tifffile.imwrite(
                str(
                    out_sub
                    / "marked"
                    / f"{Path(wf_files[i]).stem}_marked.tif"
                ),
                marked_rgb,
            )

            if len(data) == 0:
                print(
                    f"    Warning: no lipid droplets detected in "
                    f"{wf_files[i]}; skipping regression plot."
                )
                continue

            x_sim = df["mean_SIM"].values
            x_qsim = df["mean_QSIM"].values
            y = df["mean_WF"].values

            slope1, inter1, r2_1 = linear_regression(x_sim, y)
            slope2, inter2, r2_2 = linear_regression(x_qsim, y)

            plt.figure(figsize=(5, 5))

            label1 = (
                f"SIM vs WF R2={r2_1:.3f}"
                if not np.isnan(r2_1)
                else "SIM vs WF R2=N/A"
            )
            label2 = (
                f"QSIM vs WF R2={r2_2:.3f}"
                if not np.isnan(r2_2)
                else "QSIM vs WF R2=N/A"
            )

            plt.scatter(x_sim, y, s=20, label=label1)
            plt.plot(
                x_sim,
                slope1 * x_sim + inter1,
                lw=1,
                color="blue",
            )

            plt.scatter(x_qsim, y, s=20, label=label2)
            plt.plot(
                x_qsim,
                slope2 * x_qsim + inter2,
                lw=1,
                color="orange",
            )

            plt.xlabel("SIM / QSIM intensity")
            plt.ylabel("WF intensity")
            plt.title(f"{algo_name} - {Path(wf_files[i]).stem}")
            plt.legend()

            all_vals = np.concatenate([x_sim, x_qsim, y])
            min_val, max_val = all_vals.min(), all_vals.max()

            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)
            plt.plot(
                [min_val, max_val],
                [min_val, max_val],
                "k--",
                alpha=0.3,
                linewidth=1,
            )

            plt.tight_layout()
            plt.savefig(
                out_sub
                / "linreg"
                / f"{Path(wf_files[i]).stem}_linreg.png",
                dpi=150,
            )
            plt.close()

        print(f"===== {algo_name} completed =====")

    print("\nAll done. Results saved to:", args.out_root)


if __name__ == "__main__":
    main(parse_args())