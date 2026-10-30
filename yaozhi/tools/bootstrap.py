# -*- coding: utf-8 -*-
"""一键自检：生成样本 → 建立特征库 → 评估 → 热工寻优 → 纹样校验。

运行：python tools/bootstrap.py
输出：data/ 样本、models/bank.npz 特征库、models/eval_report.json 评估报告
"""
import sys
import json
import time
from pathlib import Path

# GBK 控制台兜底：无法编码的字符（如 m²）降级为 ?，避免 UnicodeEncodeError 中断自检
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                              # noqa: E402

from core import synth, kiln, pattern                          # noqa: E402
from core.defect import GlazeInspector                          # noqa: E402

TRAIN_PER_FAMILY = 8
TEST_PER_FAMILY = 3
DEFECT_PER_TYPE = 4


def gen_dataset():
    data = ROOT / "data"
    train_dir = data / "normal"
    test_dir = data / "test_normal"
    defect_dir = data / "defect"
    for d in (train_dir, test_dir, defect_dir):
        d.mkdir(parents=True, exist_ok=True)

    n_train = 0
    for fi, fam in enumerate(synth.GLAZE_FAMILIES):
        out = train_dir / fam[0]
        out.mkdir(exist_ok=True)
        for k in range(TRAIN_PER_FAMILY):
            img, _ = synth.normal_tile(512, family=fam, seed=1000 * (fi + 1) + k)
            synth.write_png(str(out / f"{fam[0]}_{k:02d}.png"), img)
            n_train += 1

    n_test = 0
    for fi, fam in enumerate(synth.GLAZE_FAMILIES):
        out = test_dir / fam[0]
        out.mkdir(exist_ok=True)
        for k in range(TEST_PER_FAMILY):
            img, _ = synth.normal_tile(512, family=fam, seed=9000 + 100 * fi + k)
            synth.write_png(str(out / f"{fam[0]}_{k:02d}.png"), img)
            n_test += 1

    n_def = 0
    for di, kind in enumerate(synth.DEFECT_TYPES):
        out = defect_dir / kind
        out.mkdir(exist_ok=True)
        for k in range(DEFECT_PER_TYPE):
            fam = synth.GLAZE_FAMILIES[(di + k) % len(synth.GLAZE_FAMILIES)]
            img, _, _ = synth.defect_tile(512, kind=kind, family=fam, seed=5000 + 100 * di + k)
            synth.write_png(str(out / f"{kind}_{k:02d}.png"), img)
            n_def += 1
    return n_train, n_test, n_def


def load_dir(p: Path):
    items = []
    fam = p.name
    for f in sorted(p.glob("*.png")):
        items.append((fam, f, synth.read_png(str(f))))
    return items


def evaluate(insp: GlazeInspector):
    test_dir = ROOT / "data" / "test_normal"
    defect_dir = ROOT / "data" / "defect"

    normal = []
    for d in sorted(test_dir.iterdir()):
        for fam, f, img in load_dir(d):
            normal.append((fam, f.name, insp.detect(img)))

    defects = []
    for d in sorted(defect_dir.iterdir()):
        for fam, f, img in load_dir(d):
            defects.append((d.name, f.name, insp.detect(img)))

    n_ok = sum(1 for _, _, r in normal if r["verdict"] == "OK")
    d_ng = sum(1 for _, _, r in defects if r["verdict"] == "NG")
    d_type = sum(1 for kind, _, r in defects if r["verdict"] == "NG" and r["defect_type"] == kind)

    per_type = {}
    for kind in synth.DEFECT_TYPES:
        rows = [r for k, _, r in defects if k == kind]
        per_type[kind] = {
            "n": len(rows),
            "ng": sum(1 for r in rows if r["verdict"] == "NG"),
            "type_hit": sum(1 for r in rows if r["verdict"] == "NG" and r["defect_type"] == kind),
            "scores": [r["score"] for r in rows],
        }

    return {
        "normal": {"n": len(normal), "ok": n_ok,
                   "ok_rate": round(100.0 * n_ok / max(1, len(normal)), 1),
                   "scores": [r["score"] for _, _, r in normal]},
        "defect": {"n": len(defects), "ng": d_ng,
                   "ng_rate": round(100.0 * d_ng / max(1, len(defects)), 1),
                   "type_hit": d_type,
                   "type_rate": round(100.0 * d_type / max(1, len(defects)), 1),
                   "per_type": per_type},
        "thresholds": insp.stats,
    }


def main():
    t0 = time.perf_counter()
    print("=" * 62)
    print("窑智 · 陶瓷工艺 AI 大脑 — 自检")
    print("=" * 62)

    print("\n[1/4] 生成样本 ...")
    n_train, n_test, n_def = gen_dataset()
    print(f"      训练良品 {n_train} 张 / 留出良品 {n_test} 张 / 缺陷 {n_def} 张")

    print("\n[2/4] 构建特征库（ResNet18 + coreset）...")
    model_path = ROOT / "models" / "bank.npz"
    model_path.parent.mkdir(exist_ok=True)
    insp = GlazeInspector(str(model_path))
    imgs = []
    for d in sorted((ROOT / "data" / "normal").iterdir()):
        for _, _, img in load_dir(d):
            imgs.append(img)
    tb = time.perf_counter()
    stats = insp.build(imgs)
    print(f"      bank={stats['bank']} 向量, 建库 {time.perf_counter() - tb:.1f}s")
    print(f"      tex_thr={stats['tex_thr']}, chroma_thr={stats['chroma_thr']}")
    print(f"      模型落盘 {model_path.stat().st_size / 1024:.0f} KB")

    print("\n[3/4] 留出集评估 ...")
    report = evaluate(insp)
    nrm, dfc = report["normal"], report["defect"]
    print(f"      良品：OK {nrm['ok']}/{nrm['n']}  准确率 {nrm['ok_rate']}%")
    print(f"      缺陷：NG {dfc['ng']}/{dfc['n']}  召回率 {dfc['ng_rate']}%  "
          f"类型命中 {dfc['type_hit']}/{dfc['n']} ({dfc['type_rate']}%)")
    for k, v in dfc["per_type"].items():
        print(f"        {k:<4} NG {v['ng']}/{v['n']}  类型命中 {v['type_hit']}/{v['n']}  "
              f"分数 {v['scores']}")

    print("\n[4/4] 热工寻优 + 纹样校验 ...")
    tk = time.perf_counter()
    opt = kiln.optimize(defect_type="釉泡", zone_bias=8)
    print(f"      釉泡/火道窑位寻优：参数网格 {opt['search_total']} 组（窑期约束内评估 {opt['search_space']} 组），{time.perf_counter() - tk:.1f}s")
    print(f"      baseline  risk={opt['baseline']['risk']}%  energy={opt['baseline']['energy_kwh_m2']}kWh/m²  "
          f"cost={opt['baseline']['total_cost']}元/m²  cycle={opt['baseline']['cycle_min']}min")
    print(f"      recommend risk={opt['recommended']['risk']}%  energy={opt['recommended']['energy_kwh_m2']}kWh/m²  "
          f"cost={opt['recommended']['total_cost']}元/m²  cycle={opt['recommended']['cycle_min']}min")
    print(f"      delta     risk {opt['delta']['risk']:+}pt  energy {opt['delta']['energy']:+}kWh/m²  "
          f"cost {opt['delta']['cost']:+}元/m²")

    pts = {}
    for style in pattern.STYLES:
        r = pattern.generate(style, "圆盘", seed=3, count=3)
        pts[style] = r["summary"]["best_score"]
        print(f"      纹样 {style:<4} 最佳可制造性 {r['summary']['best_score']}")

    report["kiln"] = {
        "baseline": opt["baseline"],
        "recommended": opt["recommended"],
        "delta": opt["delta"],
        "search_space": opt["search_space"],
    }
    report["pattern"] = pts
    report["elapsed_s"] = round(time.perf_counter() - t0, 1)
    out = ROOT / "models" / "eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n评估报告：{out}")
    print(f"总耗时 {report['elapsed_s']}s")

    ok = nrm["ok_rate"] >= 86.7 and dfc["ng_rate"] >= 80.0
    print("\n自检结论：" + ("通过（可进入前端联调）" if ok else "未达阈值，需要调参"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
