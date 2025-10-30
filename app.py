# app.py
# Simple modern UI for Old Photo Restoration
# Uses your existing restoration_src modules

from pathlib import Path
import io
import zipfile
import time

import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt

# ==== import your project building blocks ====
from restoration_src.filters_restore import median_denoise, gaussian_denoise
from restoration_src.enhance_restore import hist_equalization_bgr, clahe_bgr
from restoration_src.edges_restore import sobel_edges, canny_edges
from restoration_src.metrics_restore import compute_psnr, compute_ssim

# ---------- small helpers ----------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def load_bgr(bytes_or_path):
    """Return BGR uint8 image from bytes or path."""
    if isinstance(bytes_or_path, (str, Path)):
        img = cv2.imread(str(bytes_or_path), cv2.IMREAD_COLOR)
        return img
    file_bytes = np.asarray(bytearray(bytes_or_path.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def resample(img, scale: float):
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    new_size = (int(w * scale), int(h * scale))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(img, new_size, interpolation=interp)

def run_pipeline(
    img_bgr,
    resample_scale=1.0,
    denoise_method="median",
    ksize=5,
    sigma=1.0,
    enhance_method="clahe",
    clip=3.0,
    grid=8,
    edges_method="none",
    t1=80,
    t2=180,
):
    # 1) resample
    work = resample(img_bgr, resample_scale)

    # 2) denoise
    if denoise_method == "median":
        work = median_denoise(work, ksize)
    elif denoise_method == "gaussian":
        work = gaussian_denoise(work, ksize, sigma)
    # 'none' -> leave as is

    # 3) enhance
    if enhance_method == "he":
        work = hist_equalization_bgr(work)
    elif enhance_method == "clahe":
        work = clahe_bgr(work, clip, (grid, grid))


    # 4) metrics vs original (before resample? for UI we compare same-size)
    # Compare against original resized to same size to keep metrics meaningful
    base_for_metrics = cv2.resize(img_bgr, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_AREA)
    psnr = compute_psnr(base_for_metrics, work)
    ssim = compute_ssim(base_for_metrics, work)

    # 5) edges (optional)
    edges_img = None
    if edges_method and edges_method != "none":
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        if edges_method == "sobel":
            edges_img = sobel_edges(gray, ksize=ksize)
        elif edges_method == "canny":
            edges_img = canny_edges(gray, t1=t1, t2=t2)

    return work, psnr, ssim, edges_img

def plot_hist(before_bgr, after_bgr):
    before_gray = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
    after_gray  = cv2.cvtColor(after_bgr,  cv2.COLOR_BGR2GRAY)

    fig = plt.figure(figsize=(7.5, 3.8))
    bins = 256
    plt.hist(before_gray.ravel(), bins=bins, range=(0, 255), alpha=0.6, label="Before")
    plt.hist(after_gray.ravel(),  bins=bins, range=(0, 255), alpha=0.6, label="After")
    plt.title("Gray-level Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    return fig

def zip_bytes(files_dict):
    """files_dict: { 'path/in/zip.png': bytes, ... }"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for inside_name, data in files_dict.items():
            z.writestr(inside_name, data)
    buf.seek(0)
    return buf

# ---------- UI ----------
st.set_page_config(page_title="Old Photo Restoration", page_icon="🧿", layout="wide")

with st.sidebar:
    st.header("⚙️ Settings")

    src_mode = st.radio("Input source", ["Upload image(s)", "Pick from input_images folder"], index=0)

    resample_scale = st.slider("Resample scale", 0.2, 2.0, 1.0, 0.1)

    st.markdown("**Denoise**")
    denoise_method = st.selectbox("Method", ["none", "median", "gaussian"], index=1)
    ksize = st.select_slider("Kernel size (odd)", options=[3,5,7,9], value=5)
    sigma = st.slider("Gaussian σ", 0.0, 3.0, 1.0, 0.1)

    st.markdown("---")
    st.markdown("**Enhance**")
    enhance_method = st.selectbox("Method", ["none", "he", "clahe"], index=2)
    clip = st.slider("CLAHE clip", 1.0, 5.0, 3.0, 0.1)
    grid = st.select_slider("CLAHE grid", options=[4,6,8,10,12], value=8)

    st.markdown("---")
    st.markdown("**Edges (optional)**")
    edges_method = st.selectbox("Edges", ["none", "sobel", "canny"], index=0)
    t1 = st.slider("Canny T1", 0, 255, 80, 1)
    t2 = st.slider("Canny T2", 0, 255, 180, 1)

    st.markdown("---")
    run_btn = st.button("🚀 Run restoration", use_container_width=True)

st.title("Old Photo Restoration – Modern UI")

# Input area
images = []
names  = []

if src_mode == "Upload image(s)":
    uploads = st.file_uploader("Upload JPG/PNG/BMP/TIFF", accept_multiple_files=True, type=list(x[1:] for x in IMG_EXTS))
    if uploads:
        images = [load_bgr(f) for f in uploads]
        names  = [f.name for f in uploads]
else:
    default_dir = Path("input_images")
    options = [str(p) for p in sorted(default_dir.glob("*")) if p.suffix.lower() in IMG_EXTS]
    pick = st.multiselect("Pick images from input_images/", options, default=options[:2] if options else [])
    if pick:
        images = [load_bgr(p) for p in pick]
        names  = [Path(p).name for p in pick]

if not images:
    st.info("Upload or pick at least one image to get started.")
    st.stop()

# Process each image on Run
if run_btn:
    results_zip = {}
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for img_bgr, name in zip(images, names):
        col1, col2 = st.columns(2, gap="large")

        # pipeline
        restored, psnr, ssim, edges_img = run_pipeline(
            img_bgr,
            resample_scale=resample_scale,
            denoise_method=denoise_method,
            ksize=ksize,
            sigma=sigma,
            enhance_method=enhance_method,
            clip=clip,
            grid=grid,
            edges_method=edges_method,
            t1=t1, t2=t2
        )

        # display
        with col1:
            st.subheader(f"📷 {name} — Before")
            st.image(to_rgb(img_bgr), use_column_width=True)

        with col2:
            st.subheader("✨ After")
            st.image(to_rgb(restored), use_column_width=True)
            st.metric("PSNR (dB)", f"{psnr:.2f}")
            st.metric("SSIM", f"{ssim:.3f}")

        # histogram
        fig = plot_hist(img_bgr, restored)
        st.pyplot(fig, use_container_width=True)

        # optional edges preview
        if edges_img is not None:
            st.caption("Edges preview")
            if edges_method == "sobel":
                # Sobel often returned in uint8 already; if not, normalize
                if edges_img.dtype != np.uint8:
                    e = cv2.normalize(edges_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                else:
                    e = edges_img
                st.image(e, clamp=True, use_column_width=True)
            else:
                st.image(edges_img, clamp=True, use_column_width=True)

        # collect files for download (PNG + CSV row in a simple TSV)
        # save PNGs to memory
        _, png_after = cv2.imencode(".png", restored)
        results_zip[f"{name.rsplit('.',1)[0]}_restored.png"] = png_after.tobytes()

        # histogram figure to PNG bytes
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        buf.seek(0)
        results_zip[f"{name.rsplit('.',1)[0]}_hist.png"] = buf.read()

        # edges
        if edges_img is not None:
            _, png_edges = cv2.imencode(".png", edges_img)
            results_zip[f"{name.rsplit('.',1)[0]}_edges_{edges_method}.png"] = png_edges.tobytes()

        # add a tiny “metrics.tsv”
        line = f"{name}\t{resample_scale}\t{denoise_method}\t{ksize}\t{sigma}\t{enhance_method}\t{clip}\t{grid}\t{edges_method}\t{t1}\t{t2}\t{psnr:.4f}\t{ssim:.4f}\n"
        prev = results_zip.get("run_log.tsv", "").encode()
        results_zip["run_log.tsv"] = (prev + (b"" if prev==b"" else b"") + line.encode())

        st.divider()

    # offer zip download
    zbuf = zip_bytes(results_zip)
    st.download_button(
        "⬇️ Download all results as ZIP",
        data=zbuf,
        file_name=f"restoration_results_{timestamp}.zip",
        mime="application/zip",
        use_container_width=True
    )
else:
    st.caption("Adjust settings in the left panel, then click **Run restoration**.")
