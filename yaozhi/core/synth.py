# -*- coding: utf-8 -*-
"""釉面样本合成：正常釉面 + 五类典型缺陷（釉泡/针孔/裂纹/落渣/色差）。

用途：在拿不到工厂实拍样本前，用于特征库自校准与离线演示；
现场接入实拍后，同一套接口可直接用真实良品图重建特征库。
所有合成参数固定随机种子，结果可复现。
"""
import numpy as np
import cv2

# BGR 顺序的釉色族：(名称, 主色, 次色)
GLAZE_FAMILIES = [
    ("影青", (216, 228, 222), (190, 206, 198)),
    ("天青", (178, 199, 192), (150, 176, 170)),
    ("霁蓝", (132, 80, 50), (98, 58, 36)),
    ("豇豆红", (122, 110, 192), (92, 80, 162)),
    ("米白", (234, 238, 240), (208, 214, 218)),
]

DEFECT_TYPES = ["釉泡", "针孔", "裂纹", "落渣", "色差"]


def _noise(h, w, scale, rng):
    small = rng.normal(0.0, 1.0, (max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def normal_tile(size=512, family=None, seed=0):
    """生成一片良品釉面。返回 (BGR 图, 釉色族名)。"""
    rng = np.random.default_rng(seed)
    if family is None:
        family = GLAZE_FAMILIES[int(rng.integers(len(GLAZE_FAMILIES)))]
    name, c1, c2 = family
    base = np.zeros((size, size, 3), np.float32) + np.array(c1, np.float32)

    # 多尺度釉面起伏（流釉、色心）
    g = (_noise(size, size, 6, rng) * 0.55
         + _noise(size, size, 20, rng) * 0.30
         + _noise(size, size, 60, rng) * 0.15)
    g = g / (np.abs(g).max() + 1e-6)
    base += g[..., None] * 11.0

    # 轻微方向性釉流
    yy, xx = np.mgrid[0:size, 0:size]
    base += ((xx / size) - 0.5)[..., None] * 3.0
    # 釉层厚度不均导致的细颗粒
    base += rng.normal(0, 1.3, (size, size, 3))
    img = np.clip(base, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.6)
    return img, name


def _bubble(img, rng):
    h, w = img.shape[:2]
    r = int(rng.integers(11, 26))
    cx, cy = int(rng.integers(r + 8, w - r - 8)), int(rng.integers(r + 8, h - r - 8))
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    ring = np.exp(-((d - r) ** 2) / (2 * (r * 0.24) ** 2))
    core = (d < r * 0.82).astype(np.float32)
    core = cv2.GaussianBlur(core, (0, 0), max(1.0, r * 0.28))
    delta = ring * 34.0 - core * 20.0
    out = np.clip(img.astype(np.float32) + delta[..., None], 0, 255).astype(np.uint8)
    return out, (cx - r, cy - r, cx + r, cy + r)


def _pinhole(img, rng):
    h, w = img.shape[:2]
    r = int(rng.integers(2, 6))
    cx, cy = int(rng.integers(12, w - 12)), int(rng.integers(12, h - 12))
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    core = np.exp(-(d ** 2) / (2 * (r + 1.2) ** 2))
    out = np.clip(img.astype(np.float32) - core[..., None] * 62.0, 0, 255).astype(np.uint8)
    return out, (cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4)


def _crack(img, rng):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    x, y = int(rng.integers(40, w - 40)), int(rng.integers(40, h - 40))
    ang = rng.uniform(0, 2 * np.pi)
    pts = [(x, y)]
    for _ in range(int(rng.integers(6, 12))):
        ang += rng.normal(0, 0.35)
        step = rng.uniform(12, 30)
        x = int(np.clip(x + np.cos(ang) * step, 6, w - 7))
        y = int(np.clip(y + np.sin(ang) * step, 6, h - 7))
        pts.append((x, y))
    for i in range(len(pts) - 1):
        cv2.line(mask, pts[i], pts[i + 1], 255, int(rng.integers(1, 3)))
    if rng.random() < 0.6 and len(pts) > 3:
        b = pts[int(rng.integers(1, len(pts) - 1))]
        cv2.line(mask, b, (int(np.clip(b[0] + rng.normal(0, 26), 0, w - 1)),
                           int(np.clip(b[1] + rng.normal(0, 26), 0, h - 1))), 255, 1)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.1) / 255.0
    out = np.clip(img.astype(np.float32) - soft[..., None] * 70.0, 0, 255).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return out, box


def _debris(img, rng):
    h, w = img.shape[:2]
    cx, cy = int(rng.integers(30, w - 30)), int(rng.integers(30, h - 30))
    axes = (int(rng.integers(7, 15)), int(rng.integers(5, 11)))
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (cx, cy), axes, float(rng.uniform(0, 180)), 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
    shadow = np.zeros((h, w), np.uint8)
    cv2.ellipse(shadow, (cx + 4, cy + 5), (axes[0] + 1, axes[1] + 1), 0, 0, 360, 255, -1)
    shadow = cv2.GaussianBlur(shadow.astype(np.float32), (0, 0), 3.0) / 255.0
    blob = mask.astype(np.float32) / 255.0
    out = np.clip(img.astype(np.float32) + blob[..., None] * 46.0 - shadow[..., None] * 26.0,
                  0, 255).astype(np.uint8)
    box = (cx - axes[0] - 6, cy - axes[1] - 6, cx + axes[0] + 6, cy + axes[1] + 6)
    return out, box


def _color_dev(img, rng):
    h, w = img.shape[:2]
    cx, cy = int(rng.integers(w // 4, w * 3 // 4)), int(rng.integers(h // 4, h * 3 // 4))
    r = int(rng.integers(70, 140))
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    m = np.exp(-(d ** 2) / (2 * (r * 0.55) ** 2))[..., None]
    tint = np.array([-14.0, 4.0, 18.0], np.float32)  # 偏黄（B 降 R 升）
    out = np.clip(img.astype(np.float32) + m * tint, 0, 255).astype(np.uint8)
    return out, (max(0, cx - r), max(0, cy - r), min(w, cx + r), min(h, cy + r))


_DRAW = {"釉泡": _bubble, "针孔": _pinhole, "裂纹": _crack, "落渣": _debris, "色差": _color_dev}


def defect_tile(size=512, kind="釉泡", family=None, seed=0):
    """生成带单一缺陷的样本。返回 (BGR 图, 釉色族名, 缺陷框)。"""
    img, name = normal_tile(size=size, family=family, seed=seed)
    rng = np.random.default_rng(seed * 7919 + 13)
    img, box = _DRAW[kind](img, rng)
    return img, name, box


def write_png(path, img):
    cv2.imencode(".png", img)[1].tofile(path)


def read_png(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
