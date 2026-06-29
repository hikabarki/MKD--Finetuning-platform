"""
pdf_utils.py — Convert PDF pages to JPEG images and build VL training JSONL files.

Standalone usage:
    python pdf_utils.py document.pdf --output-dir ./pages
    python pdf_utils.py document.pdf --output-dir ./pages --jsonl  # also write JSONL
"""

import argparse
import json
import os
import sys


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 150) -> list[str]:
    """
    Convert each page of a PDF to a JPEG in output_dir.

    Tries pdf2image (poppler) first; falls back to a pypdf-based approach.
    Returns a sorted list of absolute image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.abspath(pdf_path)
    output_dir = os.path.abspath(output_dir)

    # --- Attempt 1: pdf2image (requires poppler) ---
    try:
        from pdf2image import convert_from_path

        pages = convert_from_path(pdf_path, dpi=dpi)
        image_paths = []
        for i, page in enumerate(pages, start=1):
            out_path = os.path.join(output_dir, f"page_{i:03d}.jpg")
            page.save(out_path, "JPEG", quality=90)
            image_paths.append(out_path)
        return image_paths

    except ImportError:
        pass  # fall through to next method
    except Exception as e:
        print(
            f"[pdf_utils] pdf2image failed: {e}. Trying fallback...",
            file=sys.stderr,
        )

    # --- Attempt 2: pypdf + Pillow (rasterises embedded images only) ---
    try:
        import pypdf
        from PIL import Image as _PILImage

        reader = pypdf.PdfReader(pdf_path)
        image_paths = []
        for page_num, page in enumerate(reader.pages, start=1):
            # Extract the first image found on this page, if any
            extracted = False
            if hasattr(page, "images") and page.images:
                for img_obj in page.images:
                    try:
                        pil_img = _PILImage.open(img_obj.image)
                        out_path = os.path.join(output_dir, f"page_{page_num:03d}.jpg")
                        pil_img.convert("RGB").save(out_path, "JPEG", quality=90)
                        image_paths.append(out_path)
                        extracted = True
                        break
                    except Exception:
                        continue

            if not extracted:
                print(
                    f"[pdf_utils] WARNING: page {page_num} — no rasterisable image "
                    f"found via pypdf fallback. Install poppler + pdf2image for full "
                    f"page rendering.",
                    file=sys.stderr,
                )

        if image_paths:
            return image_paths

        raise RuntimeError(
            "pypdf fallback produced no images. "
            "Install poppler and pdf2image for reliable PDF rendering:\n"
            "  sudo apt-get install poppler-utils\n"
            "  pip install pdf2image"
        )

    except ImportError:
        raise RuntimeError(
            "Neither pdf2image nor pypdf is available.\n"
            "Install one of:\n"
            "  pip install pdf2image  (also needs: sudo apt-get install poppler-utils)\n"
            "  pip install pypdf Pillow"
        )


# ---------------------------------------------------------------------------
# JSONL builder
# ---------------------------------------------------------------------------

def pdf_dataset_to_jsonl(
    pdf_path: str,
    dataset_dir: str,
    prompt: str = "Describe this page.",
) -> str:
    """
    Convert a PDF to a VL training JSONL file.

    Each page becomes one image-text pair:
        {"conversations": [{"role": "user", "content": "<prompt>"},
                           {"role": "assistant", "content": ""}],
         "image": "/abs/path/to/page_001.jpg"}

    Images are written to <dataset_dir>/images/.
    Returns the absolute path to <dataset_dir>/normalized.jsonl.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    images_dir = os.path.join(dataset_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"[pdf_utils] Rasterising {pdf_path} → {images_dir} …", flush=True)
    image_paths = pdf_to_images(pdf_path, images_dir)
    print(f"[pdf_utils] {len(image_paths)} pages converted.", flush=True)

    jsonl_path = os.path.join(dataset_dir, "normalized.jsonl")
    with open(jsonl_path, "w") as f:
        for img_path in image_paths:
            record = {
                "conversations": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ""},
                ],
                "image": img_path,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[pdf_utils] JSONL written to: {jsonl_path}", flush=True)
    print(
        "\n⚠  WARNING: All assistant labels are empty strings.\n"
        "   Edit the output JSONL and fill in the 'content' field for each\n"
        "   assistant turn before training, or training will produce a broken model.\n"
        f"   File: {jsonl_path}\n",
        flush=True,
    )
    return jsonl_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a PDF to JPEG images (and optionally a VL training JSONL)"
    )
    parser.add_argument("pdf", help="Path to the input PDF file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output images (default: <pdf_basename>_pages/)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Rendering resolution in DPI (default: 150)",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Also write a normalized.jsonl training file alongside the images",
    )
    parser.add_argument(
        "--prompt",
        default="Describe this page.",
        help="User-turn prompt for each page in the JSONL (default: 'Describe this page.')",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    pdf_path = os.path.abspath(args.pdf)
    if not os.path.isfile(pdf_path):
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(os.path.dirname(pdf_path), f"{base}_pages")

    if args.jsonl:
        jsonl_path = pdf_dataset_to_jsonl(
            pdf_path=pdf_path,
            dataset_dir=output_dir,
            prompt=args.prompt,
        )
        print(f"Done. JSONL: {jsonl_path}")
    else:
        image_paths = pdf_to_images(pdf_path, output_dir=output_dir, dpi=args.dpi)
        print(f"Done. {len(image_paths)} images written to: {output_dir}")
        for p in image_paths:
            print(f"  {p}")
