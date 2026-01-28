import os
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open-vocabulary segmentation evaluation")

    parser.add_argument(
        "-dataset",
        type=str, 
        default=None,
        help="Dataset where compute descriptors"
    )
    
    args = parser.parse_args()

    DATASET = args.dataset
    folder = os.path.join("./testing_view",DATASET,"view_test")

    results = []

    # Gather indices by scanning available GT files
    indices = sorted([
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(folder) if f.startswith("gt_")
    ])

    for i in indices:
        gt_path = os.path.join(folder, f"gt_{i}.JPEG")
        gs_path = os.path.join(folder, f"gs_{i}.JPEG")
        pred_path = os.path.join(folder, f"pred_{i}.JPEG")

        # Read images
        gt = imread(gt_path)
        gs = imread(gs_path)
        pred = imread(pred_path)

        # SSIM assumes grayscale OR multichannel → let it detect automatically
        ssim_gs = ssim(gt, gs, data_range=gs.max() - gs.min(), channel_axis=-1 if gt.ndim == 3 else None)
        ssim_pred = ssim(gt, pred, data_range=pred.max() - pred.min(), channel_axis=-1 if gt.ndim == 3 else None)

        # PSNR
        psnr_gs = psnr(gt, gs, data_range=gs.max() - gs.min())
        psnr_pred = psnr(gt, pred, data_range=pred.max() - pred.min())

        results.append((i, ssim_gs, psnr_gs, ssim_pred, psnr_pred))

        print(
            f"[{i}]  SSIM(gt,gs)={ssim_gs:.4f}, PSNR(gt,gs)={psnr_gs:.2f}  "
            f"SSIM(gt,pred)={ssim_pred:.4f}, PSNR(gt,pred)={psnr_pred:.2f}"
        )



    ssim_gs_list = [r[1] for r in results]
    psnr_gs_list = [r[2] for r in results]
    ssim_pred_list = [r[3] for r in results]
    psnr_pred_list = [r[4] for r in results]

    print("\n=== MEAN METRICS ===")
    print(f"Mean SSIM (GT–GS):   {np.mean(ssim_gs_list):.4f}")
    print(f"Mean PSNR (GT–GS):   {np.mean(psnr_gs_list):.2f} dB")
    print(f"Mean SSIM (GT–PRED): {np.mean(ssim_pred_list):.4f}")
    print(f"Mean PSNR (GT–PRED): {np.mean(psnr_pred_list):.2f} dB")