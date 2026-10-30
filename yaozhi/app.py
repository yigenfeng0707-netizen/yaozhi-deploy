# -*- coding: utf-8 -*-
"""窑智 · 陶瓷工艺 AI 大脑 — FastAPI 服务。

三个可离线运行的能力：
1) 釉面缺陷检测（无监督 PatchCore 轻量版，现场可校准）
2) 窑温数字孪生与曲线寻优（一维瞬态热传导）
3) 纹样生成与可制造性校验（参数化纹样 + 工艺约束闸门）

启动：python -m uvicorn app:app --host 0.0.0.0 --port 8000
"""
import os
import io
import sys
import time
import json
import base64
import shutil
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.defect import GlazeInspector                      # noqa: E402
from core import kiln, pattern, synth                        # noqa: E402

MODEL_PATH = ROOT / "models" / "bank.npz"
STATIC_DIR = ROOT / "static"
DATA_DIR = ROOT / "data"

inspector = GlazeInspector(str(MODEL_PATH))
app = FastAPI(title="窑智 · 陶瓷工艺 AI 大脑", version="1.0.0")


# ---------------------------------------------------------------- 数据模型
class KilnParams(BaseModel):
    rate1: float = 120.0
    rate2: float = 100.0
    soak_temp: float = 1180.0
    soak_min: float = 45.0
    cool_rate: float = 120.0
    hold573_min: float = 0.0


class OptimizeReq(BaseModel):
    defect_type: str | None = None
    u: float | None = None
    v: float | None = None
    zone_bias: float = 0.0
    params: KilnParams | None = None
    limit_h: float = 20.0


class DemoReq(BaseModel):
    kind: str = "釉泡"
    seed: int = 5


# ---------------------------------------------------------------- 工具
def _decode(raw: bytes):
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "无法解码图片，请上传 png/jpg")
    return img


def _png(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "编码失败")
    return Response(content=buf.tobytes(), media_type="image/png")


def _loop_info(regions, w, h):
    """缺陷框中心 → 窑位/温区 → 规则化参数建议。"""
    if not regions:
        return None
    x1, y1, x2, y2 = regions[0]["bbox"]
    u = ((x1 + x2) / 2.0) / max(1, w)
    v = ((y1 + y2) / 2.0) / max(1, h)
    zone = kiln.zone_of(u, v)
    params = kiln.defect_to_params(regions[0]["label"], zone["bias"])
    return {"zone": zone, "rule_params": params}


def _data_url_png(img):
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


# ---------------------------------------------------------------- 基础
@app.get("/api/health")
def health():
    return {"ok": True, "ready": inspector.ready, "ts": int(time.time())}


@app.get("/api/status")
def status():
    counts = {}
    if DATA_DIR.exists():
        for sub in ("normal", "defect"):
            p = DATA_DIR / sub
            if not p.exists():
                continue
            counts[sub] = sum(1 for _ in p.rglob("*.png"))
    return {
        "ready": inspector.ready,
        "model_path": str(MODEL_PATH),
        "model_size_kb": round(MODEL_PATH.stat().st_size / 1024, 1) if MODEL_PATH.exists() else 0,
        "stats": inspector.stats,
        "data_counts": counts,
        "capabilities": {
            "defect": "无监督 PatchCore 轻量版（ResNet18 中层特征 + coreset 记忆库）",
            "kiln": "一维瞬态热传导隐式差分 + 综合成本寻优",
            "pattern": "参数化纹样 + 可制造性校验",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
    }


# ---------------------------------------------------------------- 检测
@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    if not inspector.ready:
        raise HTTPException(503, "特征库未建立，请先运行 python tools/bootstrap.py")
    raw = await file.read()
    img = _decode(raw)
    t0 = time.perf_counter()
    r = inspector.detect(img)
    r["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    r["loop"] = _loop_info(r["regions"], img.shape[1], img.shape[0])
    return r


@app.post("/api/calibrate")
async def calibrate(files: list[UploadFile] = File(...)):
    """现场校准：上传 20~40 张同一釉色族的良品实拍图，重建特征库。"""
    if len(files) < 6:
        raise HTTPException(400, "至少需要 6 张良品图（建议 20~40 张）")
    imgs = [_decode(await f.read()) for f in files]
    if MODEL_PATH.exists():
        try:
            shutil.copy2(MODEL_PATH, MODEL_PATH.with_name("bank.prev.npz"))
        except OSError:
            pass
    t0 = time.perf_counter()
    stats = inspector.build(imgs)
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    return {"ok": True, "stats": stats}


@app.get("/api/sample")
def sample(kind: str = "正常", seed: int = 5, family: str = ""):
    fam = None
    for f in synth.GLAZE_FAMILIES:
        if f[0] == family:
            fam = f
            break
    if kind == "正常":
        img, _ = synth.normal_tile(512, family=fam, seed=seed)
    else:
        if kind not in synth.DEFECT_TYPES:
            raise HTTPException(400, f"未知缺陷类型：{kind}")
        img, _, _ = synth.defect_tile(512, kind=kind, family=fam, seed=seed)
    return _png(img)


# ---------------------------------------------------------------- 窑炉
@app.get("/api/kiln/simulate")
def kiln_simulate(rate1: float = 120, rate2: float = 100, soak_temp: float = 1180,
                  soak_min: float = 45, cool_rate: float = 120, hold573_min: float = 0):
    p = {"rate1": rate1, "rate2": rate2, "soak_temp": soak_temp,
         "soak_min": soak_min, "cool_rate": cool_rate, "hold573_min": hold573_min}
    return kiln.simulate(p)


@app.post("/api/kiln/optimize")
def kiln_optimize(req: OptimizeReq):
    base = req.params.dict() if req.params else dict(kiln.BASE)
    zone = kiln.zone_of(req.u, req.v) if (req.u is not None and req.v is not None) else None
    bias = zone["bias"] if zone else req.zone_bias
    defect = req.defect_type if req.defect_type in ("釉泡", "针孔", "裂纹", "落渣", "色差") else None

    base_sim = kiln.simulate(base)
    rule_params = kiln.defect_to_params(defect, bias) if defect else None
    rule_sim = kiln.simulate(rule_params) if rule_params else None
    best = kiln.optimize(base=base, defect_type=defect, zone_bias=bias, limit_h=req.limit_h)
    return {
        "zone": zone,
        "defect_type": defect,
        "baseline": base_sim,
        "rule_params": rule_params,
        "rule_sim": rule_sim,
        "recommended": best["recommended"],
        "delta": best["delta"],
        "top": best["top"],
        "search_space": best["search_space"],
        "search_total": best["search_total"],
    }


# ---------------------------------------------------------------- 纹样
@app.get("/api/pattern/generate")
def pattern_generate(style: str = "冰裂纹", vessel: str = "圆盘",
                     seed: int = 1, count: int = 4):
    count = max(1, min(6, count))
    return pattern.generate(style=style, vessel=vessel, seed=seed, count=count)


@app.get("/api/pattern/options")
def pattern_options():
    return {"styles": pattern.STYLES, "vessels": pattern.VESSELS,
            "palettes": [p[0] for p in pattern.PALETTES],
            "mm_per_px": round(pattern.MM_PER_PX, 4), "size": pattern.SIZE}


# ---------------------------------------------------------------- 闭环演示
@app.post("/api/demo")
def demo(req: DemoReq):
    """一键闭环（仿真数据，界面会明确标注）：
    缺陷样本 → 检测定位 → 窑位/温区 → 参数寻优 → 下一窑首件复检。
    """
    if not inspector.ready:
        raise HTTPException(503, "特征库未建立，请先运行 python tools/bootstrap.py")
    if req.kind not in synth.DEFECT_TYPES:
        raise HTTPException(400, f"未知缺陷类型：{req.kind}")

    t0 = time.perf_counter()
    bad, fam_name, _ = synth.defect_tile(512, kind=req.kind, seed=req.seed)
    fam = next((f for f in synth.GLAZE_FAMILIES if f[0] == fam_name), None)
    before = inspector.detect(bad)
    loop = _loop_info(before["regions"], 512, 512)
    zone = loop["zone"] if loop else kiln.zone_of(0.5, 0.5)

    opt = kiln.optimize(base=dict(kiln.BASE), defect_type=before["defect_type"],
                        zone_bias=zone["bias"])
    rule = kiln.defect_to_params(before["defect_type"], zone["bias"])
    rule_sim = kiln.simulate(rule)

    # 下一窑首件：仿真良品（更换随机种子），用同一检测器复检
    candidates = []
    for k in range(6):
        good, _ = synth.normal_tile(512, family=fam, seed=9000 + req.seed * 31 + k)
        r = inspector.detect(good)
        candidates.append((r["raw_score"], good, r))
    candidates.sort(key=lambda x: x[0])
    _, good, after = candidates[0]

    return {
        "sample_kind": req.kind,
        "family": fam_name,
        "simulated": True,
        "before": {**before, "image": _data_url_png(bad)},
        "after": {**after, "image": _data_url_png(good)},
        "zone": zone,
        "optimize": {
            "baseline": opt["baseline"],
            "recommended": opt["recommended"],
            "rule_sim": rule_sim,
            "delta": opt["delta"],
            "search_space": opt["search_space"],
            "search_total": opt["search_total"],
        },
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


# ---------------------------------------------------------------- 静态
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
