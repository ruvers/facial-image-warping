from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT_DIR / "assets" / "accessories"
SCALE = 4


def _canvas(width: int, height: int) -> np.ndarray:
    return np.zeros((height * SCALE, width * SCALE, 4), dtype=np.uint8)


def _pt(x: float, y: float) -> tuple[int, int]:
    return int(round(x * SCALE)), int(round(y * SCALE))


def _color(b: int, g: int, r: int, a: int = 255) -> tuple[int, int, int, int]:
    return b, g, r, a


def _downsample(img: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def _save(path: Path, img: np.ndarray, force: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return f"skipped {path.relative_to(ROOT_DIR)}"
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Could not write {path}")
    return f"wrote {path.relative_to(ROOT_DIR)}"


def _ellipse(img: np.ndarray, center, axes, angle, color, thickness=-1) -> None:
    cv2.ellipse(
        img,
        _pt(*center),
        (int(round(axes[0] * SCALE)), int(round(axes[1] * SCALE))),
        angle,
        0,
        360,
        color,
        thickness if thickness < 0 else max(1, int(round(thickness * SCALE))),
        cv2.LINE_AA,
    )


def _line(img: np.ndarray, p1, p2, color, thickness=1.0) -> None:
    cv2.line(
        img,
        _pt(*p1),
        _pt(*p2),
        color,
        max(1, int(round(thickness * SCALE))),
        cv2.LINE_AA,
    )


def _poly(img: np.ndarray, pts, color, fill=True, thickness=1.0) -> None:
    arr = np.array([_pt(x, y) for x, y in pts], dtype=np.int32)
    if fill:
        cv2.fillPoly(img, [arr], color, cv2.LINE_AA)
    else:
        cv2.polylines(
            img,
            [arr],
            True,
            color,
            max(1, int(round(thickness * SCALE))),
            cv2.LINE_AA,
        )


def _round_rect(img: np.ndarray, x: float, y: float, w: float, h: float, radius: float, color, thickness=-1) -> None:
    x1, y1 = _pt(x, y)
    x2, y2 = _pt(x + w, y + h)
    r = int(round(radius * SCALE))
    t = thickness if thickness < 0 else max(1, int(round(thickness * SCALE)))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, t, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, t, cv2.LINE_AA)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, t, cv2.LINE_AA)


def glasses_thin_black_round() -> np.ndarray:
    w, h = 960, 320
    img = _canvas(w, h)
    lens = _color(210, 225, 235, 42)
    frame = _color(16, 17, 19, 235)
    highlight = _color(255, 255, 255, 62)
    _ellipse(img, (305, 160), (116, 92), 0, lens)
    _ellipse(img, (655, 160), (116, 92), 0, lens)
    _ellipse(img, (305, 160), (116, 92), 0, frame, 8)
    _ellipse(img, (655, 160), (116, 92), 0, frame, 8)
    _line(img, (421, 158), (539, 158), frame, 7)
    _line(img, (420, 154), (540, 154), _color(80, 82, 86, 180), 2)
    _line(img, (188, 150), (82, 118), frame, 7)
    _line(img, (772, 150), (878, 118), frame, 7)
    _line(img, (268, 110), (324, 86), highlight, 4)
    _line(img, (620, 110), (676, 86), highlight, 4)
    return _downsample(img, w, h)


def glasses_gold_aviator() -> np.ndarray:
    w, h = 960, 340
    img = _canvas(w, h)
    lens = _color(86, 102, 110, 78)
    gold = _color(44, 164, 224, 245)
    dark_gold = _color(22, 104, 156, 230)
    left = [(205, 118), (280, 82), (397, 100), (414, 196), (360, 278), (245, 260)]
    right = [(755, 118), (680, 82), (563, 100), (546, 196), (600, 278), (715, 260)]
    _poly(img, left, lens)
    _poly(img, right, lens)
    _poly(img, left, gold, fill=False, thickness=7)
    _poly(img, right, gold, fill=False, thickness=7)
    _line(img, (399, 126), (561, 126), gold, 6)
    _line(img, (407, 158), (553, 158), dark_gold, 4)
    _line(img, (205, 118), (80, 90), gold, 7)
    _line(img, (755, 118), (880, 90), gold, 7)
    _line(img, (260, 112), (354, 98), _color(255, 255, 255, 65), 4)
    _line(img, (606, 98), (700, 112), _color(255, 255, 255, 65), 4)
    return _downsample(img, w, h)


def glasses_clear_square() -> np.ndarray:
    w, h = 960, 320
    img = _canvas(w, h)
    frame = _color(235, 238, 242, 105)
    edge = _color(175, 192, 204, 155)
    lens = _color(238, 248, 255, 34)
    _round_rect(img, 185, 82, 250, 158, 35, lens)
    _round_rect(img, 525, 82, 250, 158, 35, lens)
    _round_rect(img, 185, 82, 250, 158, 35, frame, 13)
    _round_rect(img, 525, 82, 250, 158, 35, frame, 13)
    _round_rect(img, 185, 82, 250, 158, 35, edge, 4)
    _round_rect(img, 525, 82, 250, 158, 35, edge, 4)
    _line(img, (435, 152), (525, 152), edge, 6)
    _line(img, (185, 128), (82, 92), edge, 6)
    _line(img, (775, 128), (878, 92), edge, 6)
    return _downsample(img, w, h)


def glasses_tortoise_rectangle() -> np.ndarray:
    w, h = 960, 320
    img = _canvas(w, h)
    lens = _color(120, 142, 152, 42)
    base = _color(34, 63, 94, 238)
    amber = _color(38, 119, 174, 205)
    _round_rect(img, 180, 88, 260, 146, 24, lens)
    _round_rect(img, 520, 88, 260, 146, 24, lens)
    _round_rect(img, 180, 88, 260, 146, 24, base, 16)
    _round_rect(img, 520, 88, 260, 146, 24, base, 16)
    for x, y, r in [(222, 112, 18), (335, 216, 22), (400, 130, 14), (560, 214, 18), (674, 112, 20), (742, 200, 16)]:
        cv2.circle(img, _pt(x, y), int(r * SCALE), amber, -1, cv2.LINE_AA)
    _line(img, (440, 151), (520, 151), base, 8)
    _line(img, (180, 126), (76, 96), base, 8)
    _line(img, (780, 126), (884, 96), base, 8)
    return _downsample(img, w, h)


def earring_gold_hoop() -> np.ndarray:
    w, h = 320, 420
    img = _canvas(w, h)
    shadow = _color(0, 0, 0, 55)
    gold = _color(36, 162, 232, 245)
    _ellipse(img, (164, 222), (82, 132), 0, shadow, 16)
    _ellipse(img, (158, 214), (82, 132), 0, gold, 15)
    _ellipse(img, (128, 140), (11, 11), 0, _color(48, 184, 246, 255))
    _line(img, (120, 126), (154, 92), _color(255, 244, 190, 125), 3)
    return _downsample(img, w, h)


def earring_silver_stud() -> np.ndarray:
    w, h = 260, 260
    img = _canvas(w, h)
    _ellipse(img, (136, 138), (56, 56), 0, _color(0, 0, 0, 45))
    _ellipse(img, (128, 128), (58, 58), 0, _color(210, 214, 216, 245))
    _ellipse(img, (108, 104), (16, 14), 0, _color(255, 255, 255, 170))
    _ellipse(img, (150, 152), (18, 16), 0, _color(125, 130, 138, 80))
    return _downsample(img, w, h)


def earring_pearl_drop() -> np.ndarray:
    w, h = 300, 460
    img = _canvas(w, h)
    gold = _color(45, 165, 226, 245)
    pearl = _color(226, 232, 238, 248)
    _ellipse(img, (150, 88), (24, 24), 0, gold)
    _line(img, (150, 112), (150, 186), gold, 4)
    _ellipse(img, (150, 268), (62, 78), 0, _color(0, 0, 0, 42))
    _ellipse(img, (144, 260), (64, 80), 0, pearl)
    _ellipse(img, (122, 224), (18, 16), -20, _color(255, 255, 255, 160))
    return _downsample(img, w, h)


def earring_small_black_ring() -> np.ndarray:
    w, h = 280, 340
    img = _canvas(w, h)
    black = _color(12, 13, 15, 240)
    _ellipse(img, (142, 178), (58, 82), 0, black, 13)
    _ellipse(img, (122, 118), (11, 11), 0, black)
    _line(img, (114, 114), (90, 92), _color(80, 84, 88, 120), 3)
    return _downsample(img, w, h)


def clip_gold_barrette() -> np.ndarray:
    w, h = 560, 180
    img = _canvas(w, h)
    _round_rect(img, 56, 62, 448, 54, 22, _color(0, 0, 0, 38))
    _round_rect(img, 48, 54, 448, 54, 22, _color(34, 154, 226, 242))
    _line(img, (86, 66), (456, 66), _color(255, 244, 190, 100), 4)
    _line(img, (92, 100), (454, 100), _color(36, 106, 154, 100), 3)
    return _downsample(img, w, h)


def clip_pearl_clip() -> np.ndarray:
    w, h = 560, 190
    img = _canvas(w, h)
    _line(img, (76, 96), (484, 96), _color(42, 135, 202, 220), 8)
    for i in range(9):
        x = 92 + i * 46
        _ellipse(img, (x + 4, 102), (20, 20), 0, _color(0, 0, 0, 34))
        _ellipse(img, (x, 96), (20, 20), 0, _color(226, 232, 238, 248))
        _ellipse(img, (x - 7, 88), (5, 5), 0, _color(255, 255, 255, 150))
    return _downsample(img, w, h)


def clip_black_pin() -> np.ndarray:
    w, h = 560, 160
    img = _canvas(w, h)
    black = _color(12, 13, 15, 242)
    _line(img, (72, 70), (480, 88), black, 9)
    _line(img, (78, 102), (470, 120), black, 6)
    _line(img, (72, 70), (78, 102), black, 7)
    _line(img, (434, 86), (470, 120), _color(70, 72, 78, 130), 4)
    return _downsample(img, w, h)


def clip_pink_clip() -> np.ndarray:
    w, h = 560, 180
    img = _canvas(w, h)
    _round_rect(img, 70, 56, 420, 64, 24, _color(0, 0, 0, 30))
    _round_rect(img, 62, 50, 420, 64, 24, _color(165, 88, 235, 235))
    _line(img, (100, 62), (440, 62), _color(240, 202, 255, 115), 5)
    _line(img, (112, 104), (432, 104), _color(102, 36, 170, 85), 4)
    return _downsample(img, w, h)


ASSETS = {
    "glasses": {
        "thin_black_round.png": glasses_thin_black_round,
        "gold_aviator.png": glasses_gold_aviator,
        "clear_square.png": glasses_clear_square,
        "tortoise_rectangle.png": glasses_tortoise_rectangle,
    },
    "earrings": {
        "gold_hoop.png": earring_gold_hoop,
        "silver_stud.png": earring_silver_stud,
        "pearl_drop.png": earring_pearl_drop,
        "small_black_ring.png": earring_small_black_ring,
    },
    "hair_clips": {
        "gold_barrette.png": clip_gold_barrette,
        "pearl_clip.png": clip_pearl_clip,
        "black_pin.png": clip_black_pin,
        "pink_clip.png": clip_pink_clip,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local transparent PNG demo accessory assets.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing generated assets.")
    args = parser.parse_args()

    for category, assets in ASSETS.items():
        for filename, factory in assets.items():
            rel = ASSET_ROOT / category / filename
            print(_save(rel, factory(), args.force))


if __name__ == "__main__":
    main()
