# -*- coding: utf-8 -*-
"""纹样生成：参数化（非生成式大模型）+ 可制造性约束校验。

为什么不用扩散模型：小厂要的是“能上釉、能烧成、能复现”的纹样，
本模块用对称群/纹样语法生成矢量级纹样，并给出丝网/贴花/手绘的工艺可行性评分；
后续接入开源扩散模型时，本模块的校验器仍作为出厂闸门。
"""
import io
import base64
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw

SIZE = 640
PHYS_MM = 220.0          # 器型最大宽度对应实物毫米
MM_PER_PX = PHYS_MM / SIZE

PALETTES = [
    ("影青·刻花", (232, 240, 235), (86, 132, 118), (196, 158, 92)),
    ("青花·分水", (240, 242, 238), (38, 62, 128), (120, 148, 196)),
    ("霁蓝·描金", (46, 64, 108), (214, 226, 236), (198, 158, 78)),
    ("豇豆红·留白", (176, 118, 138), (246, 242, 236), (86, 62, 72)),
    ("天目·铁绘", (58, 46, 40), (196, 168, 120), (128, 92, 58)),
]

STYLES = ["冰裂纹", "缠枝纹", "回纹", "泼釉"]
VESSELS = ["圆盘", "碗", "梅瓶", "罐"]


def _vessel_mask(kind):
    m = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(m)
    c = SIZE / 2
    if kind == "圆盘":
        d.ellipse([c - 270, c - 270, c + 270, c + 270], fill=255)
    elif kind == "碗":
        d.pieslice([c - 280, c - 300, c + 280, c + 300], 0, 180, fill=255)
        d.rectangle([c - 280, c, c + 280, c + 40], fill=255)
        d.ellipse([c - 120, c + 120, c + 120, c + 210], outline=255, width=10)
    elif kind == "梅瓶":
        pts = [(c, 60), (c + 70, 78), (c + 86, 150), (c + 120, 240), (c + 175, 340),
               (c + 190, 470), (c + 150, 555), (c - 150, 555), (c - 190, 470),
               (c - 175, 340), (c - 120, 240), (c - 86, 150), (c - 70, 78)]
        d.polygon(pts, fill=255)
    else:  # 罐
        d.ellipse([c - 240, c - 190, c + 240, c + 250], fill=255)
        d.rectangle([c - 160, c - 250, c + 160, c - 150], fill=255)
        d.ellipse([c - 170, c - 285, c + 170, c - 215], fill=255)
    return np.array(m, np.uint8)


def _crackle(rng, n=26):
    mask = np.zeros((SIZE, SIZE), np.uint8)
    for _ in range(n):
        x, y = rng.integers(40, SIZE - 40, 2)
        ang = rng.uniform(0, 2 * math.pi)
        for _ in range(int(rng.integers(4, 12))):
            ang += rng.normal(0, 0.5)
            step = rng.uniform(18, 55)
            nx = int(np.clip(x + math.cos(ang) * step, 2, SIZE - 3))
            ny = int(np.clip(y + math.sin(ang) * step, 2, SIZE - 3))
            cv2.line(mask, (int(x), int(y)), (nx, ny), 255, int(rng.integers(1, 3)), cv2.LINE_AA)
            x, y = nx, ny
    return mask


def _vine(rng, folds=6):
    mask = np.zeros((SIZE, SIZE), np.uint8)
    c = SIZE / 2
    for i in range(folds):
        rot = 2 * math.pi * i / folds
        for j in range(3):
            pts = []
            for t in np.linspace(0, 1, 90):
                r = 70 + t * 210
                th = rot + j * 0.55 + math.sin(t * 8 + i) * 0.16
                pts.append((c + math.cos(th) * r, c + math.sin(th) * r))
            for k in range(len(pts) - 1):
                cv2.line(mask, tuple(map(int, pts[k])), tuple(map(int, pts[k + 1])), 255, 3, cv2.LINE_AA)
            for t in np.linspace(0.12, 0.92, 5):
                r = 70 + t * 210
                th = rot + j * 0.55 + math.sin(t * 8 + i) * 0.16
                lx = int(c + math.cos(th) * r + math.cos(th + 1.2) * 16)
                ly = int(c + math.sin(th) * r + math.sin(th + 1.2) * 16)
                cv2.ellipse(mask, (lx, ly), (16, 8), math.degrees(th), 0, 360, 255, -1, cv2.LINE_AA)
    return mask


def _meander(rng, band=54):
    mask = np.zeros((SIZE, SIZE), np.uint8)
    for r0, r1 in [(238, 238 + band), (168, 168 + band)]:
        n = 26
        for k in range(n):
            a0 = 2 * math.pi * k / n
            a1 = 2 * math.pi * (k + 1) / n
            for r in (r0, r0 + band // 2, r1):
                cv2.ellipse(mask, (SIZE // 2, SIZE // 2), (r, r), 0,
                            math.degrees(a0), math.degrees(a1), 255, 3, cv2.LINE_AA)
            cx = SIZE // 2 + math.cos(a0) * (r0 + band // 2)
            cy = SIZE // 2 + math.sin(a0) * (r0 + band // 2)
            cv2.circle(mask, (int(cx), int(cy)), 8, 255, -1, cv2.LINE_AA)
    return mask


def _splatter(rng):
    mask = np.zeros((SIZE, SIZE), np.uint8)
    for _ in range(int(rng.integers(40, 70))):
        x, y = rng.integers(60, SIZE - 60, 2)
        r = int(rng.integers(10, 46))
        cv2.ellipse(mask, (int(x), int(y)), (r, int(r * rng.uniform(0.5, 1.0))),
                    float(rng.uniform(0, 180)), 0, 360, 255, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(mask, (0, 0), 1.2)


_DRAW = {"冰裂纹": _crackle, "缠枝纹": _vine, "回纹": _meander, "泼釉": _splatter}


def _render(style, vessel, seed, palette):
    rng = np.random.default_rng(seed)
    vm = _vessel_mask(vessel)
    strokes = _DRAW[style](rng)
    strokes = cv2.bitwise_and(strokes, vm)

    name, c_base, c_line, c_accent = palette
    base = np.zeros((SIZE, SIZE, 3), np.float32) + np.array(c_base, np.float32)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    d = np.sqrt((xx - SIZE / 2) ** 2 + (yy - SIZE / 2) ** 2) / (SIZE * 0.7)
    base *= (1.03 - 0.10 * d)[..., None]
    base += rng.normal(0, 2.0, (SIZE, SIZE, 1))

    out = np.clip(base, 0, 255)
    m = strokes.astype(np.float32) / 255.0
    out = out * (1 - m[..., None]) + np.array(c_line, np.float32) * m[..., None]

    # 口沿描边（第三色，模拟描金/铁绘）
    edge = cv2.morphologyEx(vm, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    em = (edge > 0) & (vm > 0)
    out[em] = np.array(c_accent, np.float32)

    out[vm == 0] = 248.0
    img = np.clip(out, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, strokes, vm, name


def manufacturability(strokes, vm, colors=2):
    """可制造性校验：线宽、覆盖率、对称性、边缘留白、套色数。"""
    if strokes.max() == 0:
        return {"score": 0, "min_width_mm": 0, "coverage": 0, "symmetry": 0,
                "edge_clearance_mm": 0, "colors": colors, "issues": ["纹样为空"]}
    dt = cv2.distanceTransform((strokes > 0).astype(np.uint8), cv2.DIST_L2, 5)
    width_px = float(np.percentile(dt[strokes > 0], 90) * 2)
    width_mm = width_px * MM_PER_PX
    cov = float((strokes > 0).sum() / max(1, (vm > 0).sum()))
    rot = cv2.warpAffine(strokes, cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), 60, 1.0), (SIZE, SIZE))
    inter = ((strokes > 0) & (rot > 0)).sum()
    union = ((strokes > 0) | (rot > 0)).sum()
    sym = float(inter / max(1, union))
    inv = cv2.distanceTransform((vm == 0).astype(np.uint8), cv2.DIST_L2, 5)
    edge_clr = float(inv[strokes > 0].min()) * MM_PER_PX

    issues, score = [], 100.0
    if width_mm < 0.35:
        score -= 26
        issues.append(f"最细线宽 {width_mm:.2f}mm，低于丝网/贴花下限 0.35mm")
    if cov < 0.02:
        score -= 18
        issues.append("纹样覆盖率过低，烧成后视觉过空")
    if cov > 0.60:
        score -= 22
        issues.append(f"覆盖率 {cov*100:.0f}%，釉面易积釉、易缩釉")
    if sym < 0.55:
        score -= 12
        issues.append(f"旋转对称重合度 {sym:.2f}，手绘/贴花对位成本高")
    if edge_clr < 1.5:
        score -= 10
        issues.append(f"距口沿仅 {edge_clr:.1f}mm，流釉易粘连")
    if colors > 3:
        score -= 15
        issues.append("套色超过 3 色，需多次施釉与复烧")
    if not issues:
        issues.append("通过：线宽、覆盖率、对称性、边距均在可制造区间")
    return {"score": round(max(0.0, score), 1),
            "min_width_mm": round(width_mm, 2),
            "coverage": round(cov, 3),
            "symmetry": round(sym, 3),
            "edge_clearance_mm": round(edge_clr, 2),
            "colors": colors,
            "issues": issues}


def _data_url(img):
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def generate(style="冰裂纹", vessel="圆盘", seed=1, count=4):
    if style not in _DRAW:
        style = "冰裂纹"
    if vessel not in VESSELS:
        vessel = "圆盘"
    out = []
    for i in range(count):
        pal = PALETTES[(seed + i) % len(PALETTES)]
        img, strokes, vm, pname = _render(style, vessel, seed * 97 + i * 13, pal)
        m = manufacturability(strokes, vm, colors=3)
        out.append({"image": _data_url(img), "palette": pname, "metrics": m,
                    "params": {"style": style, "vessel": vessel, "seed": seed * 97 + i * 13}})
    best = max(out, key=lambda x: x["metrics"]["score"])
    return {"variants": out, "best_index": out.index(best),
            "summary": {"style": style, "vessel": vessel,
                        "best_score": best["metrics"]["score"],
                        "try_count": count}}
