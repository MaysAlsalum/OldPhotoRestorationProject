import argparse, csv
from pathlib import Path
import cv2, numpy as np

from .filters_restore import median_denoise, gaussian_denoise
from .enhance_restore import hist_equalization_bgr, clahe_bgr
from .edges_restore import sobel_edges, canny_edges
from .metrics_restore import compute_psnr, compute_ssim
from .visual_restore import save_before_after, save_histograms, save_edges_preview

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def load_images(path):
    p = Path(path)
    if p.is_dir():
        return sorted([x for x in p.rglob('*') if x.suffix.lower() in IMG_EXTS])
    return [p] if p.suffix.lower() in IMG_EXTS else []

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def denoise(img, method, **kw):
    if method == 'median':   return median_denoise(img, kw.get('ksize', 3))
    if method == 'gaussian': return gaussian_denoise(img, kw.get('ksize', 5), kw.get('sigma', 0.0))
    if method == 'none':     return img
    raise ValueError(f'Unknown denoise method: {method}')

def enhance(img, method, **kw):
    if method == 'he':    return hist_equalization_bgr(img)
    if method == 'clahe': return clahe_bgr(img, kw.get('clip', 3.0), kw.get('grid', (8, 8)))
    if method == 'none':  return img
    raise ValueError(f'Unknown enhancement method: {method}')

def detect_edges(gray, method, **kw):
    if method == 'sobel': return sobel_edges(gray, kw.get('ksize', 3))
    if method == 'canny': return canny_edges(gray, kw.get('t1', 100), kw.get('t2', 200))
    if method == 'none':  return None
    raise ValueError(f'Unknown edge method: {method}')

def process_one(img_path: Path, out_dir: Path, args):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Can't read {img_path}"); return

    # Optional resample
    if args.resample != 1.0:
        h, w = img.shape[:2]
        new_size = (max(1, int(w * args.resample)), max(1, int(h * args.resample)))
        interp = cv2.INTER_CUBIC if args.resample > 1.0 else cv2.INTER_AREA
        img = cv2.resize(img, new_size, interpolation=interp)

    # 1) Denoise -> 2) Enhance
    denoised  = denoise(img, args.denoise, ksize=args.ksize, sigma=args.sigma)
    enhanced  = enhance(denoised, args.enhance, clip=args.clip, grid=(args.grid, args.grid))

    # Output folders
    ensure_dir(out_dir / 'figures'); ensure_dir(out_dir / 'images')

    # Save outputs
    save_before_after(img, enhanced, out_dir / 'figures' / f'{img_path.stem}_before_after.png')
    save_histograms(img, enhanced, out_dir / 'figures' / f'{img_path.stem}_hist.png')
    cv2.imwrite(str(out_dir / 'images' / f'{img_path.stem}_restored.png'), enhanced)

    # Edges (optional)
    if args.edges != 'none':
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        e = detect_edges(gray, args.edges, ksize=args.ksize, t1=args.t1, t2=args.t2)
        if e is not None:
            save_edges_preview(e, out_dir / 'figures' / f'{img_path.stem}_edges_{args.edges}.png',
                               title=f'Edges: {args.edges}')

    # Metrics + CSV logging
    try:
        p = compute_psnr(img, enhanced); s = compute_ssim(img, enhanced)
        print(f'✅ {img_path.name}: PSNR={p:.2f} dB | SSIM={s:.4f}')
    except Exception as ex:
        p, s = None, None
        print(f'⚠️ Metrics failed for {img_path.name}: {ex}')

    with open(out_dir / 'run_log.csv', 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([
            img_path.name, args.resample, args.denoise, args.ksize, args.sigma,
            args.enhance, args.clip, args.grid, args.edges, args.t1, args.t2, p, s
        ])

def main():
    ap = argparse.ArgumentParser(description='Old Photo Restoration Tool')
    ap.add_argument('--input', required=True, help='image file or folder')
    ap.add_argument('--output', default='output_results', help='output folder')
    ap.add_argument('--resample', type=float, default=1.0, help='resize ratio (0.5=half, 2=double)')

    ap.add_argument('--denoise',  choices=['median', 'gaussian', 'none'], default='median')
    ap.add_argument('--ksize',    type=int,   default=5)
    ap.add_argument('--sigma',    type=float, default=0.0)

    ap.add_argument('--enhance',  choices=['he', 'clahe', 'none'], default='clahe')
    ap.add_argument('--clip',     type=float, default=3.0)
    ap.add_argument('--grid',     type=int,   default=8)

    ap.add_argument('--edges',    choices=['sobel', 'canny', 'none'], default='none')
    ap.add_argument('--t1',       type=int,   default=100)
    ap.add_argument('--t2',       type=int,   default=200)

    args = ap.parse_args()

    out_dir = Path(args.output); ensure_dir(out_dir)
    with open(out_dir / 'run_log.csv', 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['filename','resample','denoise','ksize','sigma','enhance','clip',
                                'grid','edges','t1','t2','psnr','ssim'])

    files = load_images(args.input)
    if not files:
        print('❌ No images found. Supported: .jpg .jpeg .png .bmp .tif .tiff'); return

    print(f'🔎 Found {len(files)} image(s)...')
    for p in files:
        process_one(p, out_dir, args)
    print(f'✅ Done! Results saved in: {out_dir}')

if __name__ == '__main__':
    main()
