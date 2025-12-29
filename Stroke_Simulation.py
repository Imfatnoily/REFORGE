import os
import sys
import numpy as np
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
import argparse
import cv2
import torch
from sympy import false

from evaluater import Evaluator


def median_filter(img: Image.Image, kernel_size: int = 23) -> Image.Image:
    img_np = np.array(img)
    kernel_size = max(3, kernel_size | 1)
    filtered = cv2.medianBlur(img_np, kernel_size)
    return Image.fromarray(filtered)


def quantize_image_adaptive(img: Image.Image, n_colors: int = 6) -> Image.Image:
    """Use KMeans to reduce image colors to `n_colors` (adaptive palette)"""
    img_np = np.array(img)
    h, w, c = img_np.shape
    img_flat = img_np.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(img_flat)
    labels = kmeans.predict(img_flat)
    quantized_flat = kmeans.cluster_centers_[labels].astype(np.uint8)
    quantized_img = quantized_flat.reshape(h, w, 3)

    return Image.fromarray(quantized_img)

def human_stroke_simulation(input_path, output_path, kernel_size=23, n_colors=6):
    os.makedirs(output_path, exist_ok=True)

    for fname in os.listdir(input_path):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(input_path, fname)
        img = Image.open(img_path).convert("RGB")

        # Step 1: median filter
        filtered_img = median_filter(img, kernel_size)
        # Step 2: adaptive color quantization
        stroke_img = quantize_image_adaptive(filtered_img, n_colors)

        # Save result
        save_path = os.path.join(output_path, fname.replace('.jpg', '.png').replace('.jpeg', '.png'))
        stroke_img.save(save_path)
        print(f"Saved stroke image to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stroke-Simulation Algorithm")
    parser.add_argument("--input_dir", default="REFORGE/dataset/reference", type=str)
    parser.add_argument("--output_dir", default="REFORGE/dataset/stroke",type=str)
    parser.add_argument("--kernel_size", type=int, default=45, help="Median filter kernel size")
    parser.add_argument("--n_colors", type=int, default=6, help="Number of adaptive colors")

    args = parser.parse_args()
    concept_list = ["style_vangogh"]

    for concept in concept_list:
        input_subdir = os.path.join(args.input_dir, concept)
        output_subdir = os.path.join(args.output_dir, concept)

        if os.path.isdir(input_subdir):
            print(f"[INFO] Processing concept: {concept}")
            os.makedirs(output_subdir, exist_ok=True)

            human_stroke_simulation(
                input_path=input_subdir,
                output_path=output_subdir,
                kernel_size=args.kernel_size,
                n_colors=args.n_colors
            )
