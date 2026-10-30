# -*- coding: utf-8 -*-
"""釉面缺陷检测内核（PatchCore 轻量化：ResNet18 中层特征 + coreset 记忆库）。

- 无监督：只用良品图建库，不需要缺陷样本标注
- 现场可校准：把 20~40 张良品实拍图 POST 进来即可重建特征库（CPU 实测 20 张约 26 秒 / 40 张约 44 秒）
- 双通道：纹理异常（深层特征距离）+ 色差异常（Lab 局部残差），覆盖“色差/流釉”
- 输出：缺陷图、异常热力图、缺陷区域框与类型、显著度分数
"""
import os
import base64
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from . import synth

# 离线优先：ResNet18 预训练权重随包携带，指向项目内权重目录
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch_hub"))

torch.set_num_threads(max(2, min(8, os.cpu_count() or 4)))
_INPUT = 224
_GRID = 14
_PATCH_THR_PCT = 99.7

DEFECT_ADVICE = {
    "釉泡": ("釉泡/气泡", "升温段过快或保温不足，坯体排气未完成即进入釉熔阶段",
             "降升温速率、延长保温 10~20min，并核对窑位局部温度"),
    "针孔": ("针孔", "釉面局部排气孔未闭合，多与升温速率、釉层厚度和窑内粉尘有关",
             "降低 900~1050℃ 升温速率，延长高温保温，检查施釉厚度一致性"),
    "裂纹": ("裂纹", "冷却段（尤其 573℃ 石英晶型转变区）温差过大产生应力",
             "降低冷却速率，573℃ 附近设 30~60min 缓冷平台"),
    "落渣": ("落渣", "窑顶/匣钵耐火材料剥落，属装窑与窑体清洁问题，非温度曲线主因",
             "停窑清渣、检查匣钵与窑顶，同步做窑位温度复核"),
    "色差": ("色差/流釉", "局部过烧或窑内温差导致釉色偏移",
             "下调保温温度、均衡窑内气流，复核火口附近窑位"),
}

_FAMILY_CACHE = {}


def _load_net():
    if "net" not in _FAMILY_CACHE:
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.eval()
        _FAMILY_CACHE["net"] = net
        for p in net.parameters():
            p.requires_grad_(False)
    return _FAMILY_CACHE["net"]


def _to_tensor(img_bgr):
    x = cv2.resize(img_bgr, (_INPUT, _INPUT), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


@torch.no_grad()
def _features(img_bgr):
    """返回 196x256 的 patch 特征与 14x14x256 特征图。"""
    net = _load_net()
    x = _to_tensor(img_bgr)
    x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
    f = net.layer2(net.layer1(x))
    f = net.layer3(f)                       # 1x256x14x14
    f = F.avg_pool2d(f, 3, stride=1, padding=1)
    feat_map = F.normalize(f[0].permute(1, 2, 0).contiguous(), p=2, dim=-1)   # 14x14x256
    return feat_map.reshape(-1, feat_map.shape[-1]).numpy(), feat_map.numpy()


def _coreset(X, k, seed=0):
    """贪心 k-center 采样，等价 PatchCore 的 coreset 缩减。"""
    n = X.shape[0]
    k = min(k, n)
    rng = np.random.default_rng(seed)
    if n > 6000:
        X = X[rng.choice(n, 6000, replace=False)]
        n = 6000
    sel = [int(rng.integers(n))]
    d = ((X - X[sel[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, ((X - X[i]) ** 2).sum(1))
    return X[np.array(sel)]


def _local_color(img_bgr):
    """Lab 多尺度局部残差（去全图偏色），返回 224x224 ΔE 图。

    细尺度（σ=9）抓针孔/落渣等小面积色偏；
    粗尺度（σ=34）抓“色差/流釉”这类大面积缓变偏色——细尺度会把它们抹平。
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = cv2.resize(lab, (_INPUT, _INPUT), interpolation=cv2.INTER_AREA)

    def _resid(sigma):
        r = lab - cv2.GaussianBlur(lab, (0, 0), sigma)
        return cv2.GaussianBlur(r, (0, 0), 2.0)

    d_fine = np.sqrt((_resid(9.0) ** 2).sum(2))
    d_coarse = np.sqrt((_resid(34.0) ** 2).sum(2))
    return np.maximum(d_fine, d_coarse), lab


def _score_map(patches, bank):
    """patch 特征 → 224x224 平滑异常分数图（与推理完全同口径）。"""
    d2 = 2.0 - 2.0 * (patches @ bank.T)
    dist = np.sqrt(np.maximum(d2, 0.0)).min(1).reshape(_GRID, _GRID)
    smap = cv2.GaussianBlur(dist, (0, 0), 1.3)
    smap = cv2.resize(smap, (_INPUT, _INPUT), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(smap, (0, 0), 4.0)


def _smooth_dE(dE):
    return cv2.GaussianBlur(dE.astype(np.float32), (0, 0), 6.0)


def _cross_peak(images):
    """两折交叉建库，取“未见良品”的纹理峰值（真实跨批次漂移上界）。"""
    idx = np.arange(len(images))
    peaks = []
    for fold in (0, 1):
        sub = [images[i] for i in idx if i % 2 == fold]
        out = [images[i] for i in idx if i % 2 != fold]
        if not sub or not out:
            continue
        pool = np.concatenate([_features(im)[0] for im in sub], 0)
        b = _coreset(pool, 1500, seed=fold).astype(np.float32)
        for im in out:
            p, _ = _features(im)
            peaks.append(float(_score_map(p, b).max()))
    return float(max(peaks)) if peaks else 0.0


def _signature(dE_raw, dE, grad, signed):
    """从异常区域的明暗/锐度/面积签名提取五类缺陷的判别量。

    合成与实拍验证结果（见 tools/bootstrap 自检）：
    - 色差：未平滑残差大连通域（big > 0.010），无锐边
    - 裂纹/针孔：深度负偏（dark < -40），裂纹连通域更大、更长
    - 釉泡/落渣：亮块偏（bright > 14），釉泡环面积大（big > 0.0015）
    """
    dmax = float(dE.max()) if dE.size else 0.0
    m = dE > 0.55 * dmax if dmax > 0 else np.zeros_like(dE, dtype=bool)
    raw = np.maximum(dE_raw, 0.0)
    tm = (raw > 0.62 * max(float(raw.max()), 1e-6)).astype(np.uint8)
    tm = cv2.morphologyEx(tm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(tm, 8)
    areas = [st[i][4] for i in range(1, n)]
    return {
        "dmax": dmax,
        "sharp": float(grad[m].mean()) if m.any() else 0.0,
        "dark": float(signed.min()),
        "bright": float(signed.max()),
        "big": (max(areas) if areas else 0) / float(_INPUT * _INPUT),
    }


def _classify(s, tex_score, chroma_score):
    """按形态签名把异常归到五类缺陷（优先级：色差 → 裂纹/针孔 → 釉泡/落渣）。"""
    big, dark, bright = s["big"], s["dark"], s["bright"]
    if big > 0.010:                                        # 大面积缓变色偏
        return "色差"
    if dark < -40.0:                                       # 深色贯入
        return "裂纹" if big >= 0.0006 else "针孔"
    if bright > 14.0:                                      # 亮块 + 硬阴影
        return "釉泡" if big > 0.0015 else "落渣"
    if big > 0.0015 and s["sharp"] < 20.0:
        return "釉泡"
    return "色差" if chroma_score >= tex_score else "釉泡"


class GlazeInspector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.bank = None
        self.patch_thr = 1.0
        self.tex_thr = 1.0
        self.img_thr = 1.0
        self.chroma_thr = 8.0
        self.stats = {}
        self.load()

    # ---------- 建库 ----------
    def build(self, images, families=None):
        patches = []
        for im in images:
            p, _ = _features(im)
            patches.append(p)
        pool = np.concatenate(patches, 0)
        bank = _coreset(pool, 3000, seed=7).astype(np.float32)
        self.bank = bank

        # 第二遍：用与推理完全相同的后处理统计“良品峰值”
        tex_peak, chroma_peak, labs = [], [], []
        for im in images:
            p, _ = _features(im)
            tex_peak.append(float(_score_map(p, bank).max()))
            dE, lab = _local_color(im)
            chroma_peak.append(float(_smooth_dE(dE).max()))
            labs.append(lab.reshape(-1, 3).mean(0))

        t_mu, t_sd = float(np.mean(tex_peak)), float(np.std(tex_peak))
        c_mu, c_sd = float(np.mean(chroma_peak)), float(np.std(chroma_peak))

        # 两折交叉建库：用“未参与建库”的良品估计跨批次漂移，避免阈值被自匹配低估
        x_peak = _cross_peak(images)
        self.tex_thr = float(max(t_mu * 1.15, x_peak * 1.08, t_mu + 3.5 * t_sd) + 1e-6)
        self.chroma_thr = float(max(c_mu + 4.0 * c_sd, c_mu * 1.35) + 1e-6)
        self.patch_thr = self.tex_thr
        self.img_thr = 1.0
        self.stats = {
            "bank": int(len(bank)),
            "train_images": len(images),
            "tex_thr": round(self.tex_thr, 4),
            "chroma_thr": round(self.chroma_thr, 3),
            "normal_peak_mu": round(t_mu, 4),
            "normal_peak_sd": round(t_sd, 4),
            "cross_peak": round(x_peak, 4),
            "chroma_peak_mu": round(c_mu, 3),
            "chroma_peak_sd": round(c_sd, 3),
        }
        np.savez_compressed(self.model_path, bank=self.bank,
                           tex_thr=self.tex_thr, chroma_thr=self.chroma_thr,
                           patch_thr=self.patch_thr, img_thr=self.img_thr,
                           family_lab=np.array(labs, np.float32))
        return self.stats

    def load(self):
        if not os.path.exists(self.model_path):
            return False
        z = np.load(self.model_path)
        self.bank = z["bank"].astype(np.float32)
        self.tex_thr = float(z["tex_thr"]) if "tex_thr" in z else float(z["patch_thr"])
        self.patch_thr = self.tex_thr
        self.img_thr = 1.0
        self.chroma_thr = float(z["chroma_thr"])
        self.stats = {"bank": int(len(self.bank)), "tex_thr": round(self.tex_thr, 4),
                      "chroma_thr": round(self.chroma_thr, 3)}
        return True

    @property
    def ready(self):
        return self.bank is not None

    # ---------- 推理 ----------
    def detect(self, img_bgr):
        if not self.ready:
            raise RuntimeError("特征库未建立，请先运行 bootstrap")
        h0, w0 = img_bgr.shape[:2]
        sx, sy = w0 / _INPUT, h0 / _INPUT
        patches, feat_map = _features(img_bgr)
        smap = _score_map(patches, self.bank)

        dE_raw, _lab = _local_color(img_bgr)
        dE = _smooth_dE(dE_raw)

        tex_score = float(smap.max()) / max(self.tex_thr, 1e-6)
        chroma_score = float(dE.max()) / max(self.chroma_thr, 1e-6)
        img_score = max(tex_score, chroma_score)
        verdict = bool(img_score > 1.0)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.resize(gray, (_INPUT, _INPUT), interpolation=cv2.INTER_AREA)
        signed = gray - cv2.GaussianBlur(gray, (0, 0), 11.0)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.abs(gx) + np.abs(gy)

        main, regions = "无", []
        if verdict:
            main = _classify(_signature(dE_raw, dE, grad, signed), tex_score, chroma_score)
            mask = ((smap > max(self.tex_thr, 0.85 * float(smap.max())))
                    | (dE > max(self.chroma_thr, 0.80 * float(dE.max())))).astype(np.uint8)
            mask = cv2.dilate(mask, np.ones((11, 11), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            n, lab_img, st, _ = cv2.connectedComponentsWithStats(mask, 8)
            for i in range(1, n):
                x, y, w, h, area = st[i]
                if area < 0.0008 * _INPUT * _INPUT:
                    continue
                sub = lab_img[y:y + h, x:x + w] == i
                c_val = float(np.percentile(dE[y:y + h, x:x + w][sub], 90))
                regions.append({
                    "label": main,
                    "bbox": [int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy)],
                    "area_ratio": round(area / (_INPUT * _INPUT), 5),
                    "confidence": round(float(min(0.99, 0.45 + 0.3 * min(
                        1.0, c_val / max(self.chroma_thr, 1e-6)))), 3),
                    "signed": round(float(signed[y:y + h, x:x + w][sub].mean()), 2),
                })
            regions.sort(key=lambda r: r["area_ratio"], reverse=True)
            regions = regions[:3]
        # 全图色差（整体偏色）单独提示
        scale = 100.0 * min(1.0, max(0.0, (img_score - 1.0) / 2.5)) if verdict else 0.0
        if not verdict:
            scale = 100.0 * min(0.35, max(0.02, img_score * 0.3))
        score = round(scale, 1)

        heat = np.clip(smap / max(smap.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
        heat = cv2.resize(heat, (w0, h0), interpolation=cv2.INTER_CUBIC)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        overlay = img_bgr.copy()
        for r in regions:
            x1, y1, x2, y2 = r["bbox"]
            color = (48, 200, 255) if r["label"] != "裂纹" else (80, 80, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, r["label"], (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        vis = cv2.addWeighted(overlay, 0.75, heat, 0.25, 0)

        kind, cause, action = DEFECT_ADVICE.get(main, ("正常", "未见显著异常", "维持当前工艺参数"))
        return {
            "ready": True,
            "verdict": "NG" if verdict else "OK",
            "score": score,
            "raw_score": round(img_score, 3),
            "threshold": 1.0,
            "regions": regions,
            "defect_type": main,
            "cause": cause,
            "action": action,
            "overlay": _data_url(vis),
            "heatmap": _data_url(heat),
            "size": [w0, h0],
        }


def _data_url(img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def build_from_dir(inspector, data_dir):
    paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    imgs = [synth.read_png(p) for p in paths]
    return inspector.build(imgs, families=[os.path.basename(os.path.dirname(p)) for p in paths])
