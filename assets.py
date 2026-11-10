"""Asset loader and CSS injection utilities for the Nocturnal Pixel-Art UI."""

import base64
import os
from pathlib import Path


def load_pixel_b64(path: str) -> str | None:
    """Reads image files from ./assets/ or the root directory and returns a data URI."""
    base_name = os.path.basename(path)
    repo_root = Path(__file__).resolve().parent.parent
    name_no_ext, ext = os.path.splitext(base_name)

    ext_candidates = [ext]
    if ext.lower() in [".jpg", ".jpeg"]:
        ext_candidates.append(".png")
    elif ext.lower() == ".png":
        ext_candidates.extend([".jpg", ".jpeg"])

    candidates: list[Path] = []
    for test_ext in ext_candidates:
        test_file = f"{name_no_ext}{test_ext}"
        candidates.extend([
            repo_root / "assets" / test_file,
            Path("assets") / test_file,
            repo_root / test_file,
            Path(test_file),
        ])

    target_path: Path | None = None
    for cand in candidates:
        if cand.is_file():
            target_path = cand
            break

    if not target_path:
        return None

    resolved_ext = target_path.suffix.lower()
    mime = "image/png"
    if resolved_ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif resolved_ext == ".webp":
        mime = "image/webp"
    elif resolved_ext == ".svg":
        mime = "image/svg+xml"

    try:
        with open(target_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


def load_css() -> str:
    """Loads ui/style.css and appends the dynamic .stApp background rule."""
    css_path = Path(__file__).resolve().parent / "style.css"
    with open(css_path, "r", encoding="utf-8") as f:
        base_css = f.read()

    clouds_b64 = load_pixel_b64("assets/clouds.jpg")
    if not clouds_b64:
        clouds_b64 = load_pixel_b64("assets/i_2.png")

    if clouds_b64:
        dynamic_bg = (
            f'.stApp {{\n'
            f'    background: linear-gradient(180deg, rgba(4, 8, 17, 0.84) 0%, rgba(5, 12, 24, 0.88) 70%, rgba(4, 9, 18, 0.95) 100%), '
            f'url("{clouds_b64}") center/cover fixed no-repeat !important;\n'
            f'}}\n'
        )
    else:
        dynamic_bg = '.stApp {\n    background-color: #040811 !important;\n}\n'

    return f"{base_css}\n{dynamic_bg}"

