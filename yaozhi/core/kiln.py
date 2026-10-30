# -*- coding: utf-8 -*-
"""窑温数字孪生：一维瞬态热传导（隐式有限差分）+ 工艺风险评估 + 曲线寻优。

说明口径（答辩必须如实讲）：
- 传热方程、边界条件、石英晶型转变区（573℃）判据是真实物理/工艺规则；
- 材料物性与换热系数取陶瓷行业常用值，尚未用某座具体窑炉实测标定，
  因此界面标注“仿真曲线”，接入工厂热电偶实测曲线后可直接反演标定。
"""
import math
import numpy as np

# 坯体物性（高白瓷/炻器常用区间）
RHO, CP, K_COND = 2200.0, 1000.0, 1.8
H_CONV = 150.0          # 综合换热系数 W/m²K（对流+辐射折算）
T_AMB = 25.0
THICK = 0.012           # 坯体厚度 m，厚胎大件取 12mm

BASE = {"rate1": 120.0, "rate2": 100.0, "soak_temp": 1180.0,
        "soak_min": 45.0, "cool_rate": 120.0, "hold573_min": 0.0}

BOUNDS = {
    "rate1": (50, 160), "rate2": (40, 140), "soak_temp": (1140, 1210),
    "soak_min": (30, 120), "cool_rate": (40, 180), "hold573_min": (0, 90),
}


def gas_profile(p):
    """按参数生成窑内气氛温度折线（分钟, ℃）。"""
    pts = [(0.0, T_AMB)]
    t = 0.0

    def seg(target, rate):
        nonlocal t
        dur = abs(target - pts[-1][1]) / max(rate, 1e-6) * 60.0
        t += dur
        pts.append((t, target))

    seg(600.0, p["rate1"])
    seg(p["soak_temp"], p["rate2"])
    t += p["soak_min"]
    pts.append((t, p["soak_temp"]))
    seg(600.0, p["cool_rate"])
    if p.get("hold573_min", 0) > 0:
        t += p["hold573_min"]
        pts.append((t, 580.0))
    seg(80.0, p["cool_rate"])
    seg(40.0, max(p["cool_rate"] * 0.3, 8.0))
    return pts


def _thomas(a, b, c, d):
    n = len(b)
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def simulate(p, dt=20.0, n=25, with_series=True):
    """隐式差分求解坯体温度场，返回曲线与工艺风险指标。"""
    pts = gas_profile(p)
    total_s = pts[-1][0] * 60.0
    steps = int(total_s / dt) + 1
    t_s = np.arange(steps) * dt
    t_min = t_s / 60.0
    gas = np.interp(t_min, [q[0] for q in pts], [q[1] for q in pts])

    dx = THICK / (n - 1)
    alpha = K_COND / (RHO * CP)
    aa = alpha * dt / dx ** 2
    bb = 2 * H_CONV * dt / (RHO * CP * dx)

    a = np.zeros(n)
    b = np.full(n, 1.0 + 2 * aa)
    c = np.zeros(n)
    for i in range(1, n - 1):
        a[i] = -aa
        c[i] = -aa
    b[0] = b[-1] = 1.0 + bb + 2 * aa
    c[0] = -2 * aa
    a[-1] = -2 * aa

    T = np.full(n, T_AMB)
    surf = np.empty(steps)
    core = np.empty(steps)
    for k in range(steps):
        d = T.copy()
        d[0] += bb * gas[k]
        d[-1] += bb * gas[k]
        T = _thomas(a, b, c, d)
        surf[k] = T[0]
        core[k] = T[n // 2]

    dT = surf - core
    heat_dt = float(dT.max())
    q_mask = (core >= 540) & (core <= 620)
    quartz_dt = float(np.abs(dT[q_mask]).max()) if q_mask.any() else 0.0

    ds = np.gradient(surf, dt)
    cool_mask = (ds < 0) & (surf >= 500) & (surf <= 650)
    cool_rate = float(np.abs(ds[cool_mask]).max() * 3600) if cool_mask.any() else 0.0

    flux = H_CONV * np.maximum(gas - surf, 0.0)
    standby = np.where(gas > 600, 0.9, 0.25) * 1000.0
    energy = float((flux.sum() * 2 + standby.sum()) * dt / 3.6e6)

    r_heat = min(1.0, heat_dt / 8.0)
    r_quartz = min(1.0, quartz_dt / 4.0)
    r_cool = min(1.0, cool_rate / 150.0)
    r_over = min(1.0, max(0.0, p["soak_temp"] - 1200.0) / 40.0)
    r_soak = min(1.0, max(0.0, 40.0 - p["soak_min"]) / 40.0)

    risk = 100.0 * (0.28 * r_heat + 0.30 * r_quartz + 0.26 * r_cool
                    + 0.08 * r_over + 0.08 * r_soak)
    warp = 100.0 * min(1.0, 0.55 * r_quartz + 0.25 * r_cool + 0.20 * r_heat)
    crack = 100.0 * min(1.0, 0.65 * r_cool + 0.35 * r_quartz)

    cycle_min = float(t_min[-1])
    energy_cost = energy / 0.85 * 0.42
    scrap_rate = risk / 100.0 * 0.35
    scrap_cost = 150.0 * scrap_rate
    total_cost = energy_cost + scrap_cost

    out = {
        "params": {k: round(float(v), 1) for k, v in p.items()},
        "cycle_min": round(cycle_min, 1),
        "energy_kwh_m2": round(energy, 2),
        "max_deltaT": round(heat_dt, 2),
        "quartz_deltaT": round(quartz_dt, 2),
        "cool_rate": round(cool_rate, 1),
        "risk": round(risk, 1),
        "warp_risk": round(warp, 1),
        "crack_risk": round(crack, 1),
        "scrap_rate": round(scrap_rate * 100, 2),
        "energy_cost": round(energy_cost, 2),
        "scrap_cost": round(scrap_cost, 2),
        "total_cost": round(total_cost, 2),
        "gas_peak": round(float(gas.max()), 1),
    }
    if with_series:
        idx = np.linspace(0, steps - 1, min(240, steps)).astype(int)
        out["series"] = {
            "t": [round(float(x), 1) for x in t_min[idx]],
            "gas": [round(float(x), 1) for x in gas[idx]],
            "surface": [round(float(x), 1) for x in surf[idx]],
            "core": [round(float(x), 1) for x in core[idx]],
        }
    return out


def _quick(p):
    return simulate(p, dt=60.0, n=13, with_series=False)


def optimize(base=None, defect_type=None, zone_bias=0.0, limit_h=24.0):
    """在参数空间内寻优，目标：综合成本（能耗 + 废品损失）最低。"""
    base = dict(base or BASE)
    grid = {"rate1": [60.0, 90.0, 120.0], "rate2": [50.0, 80.0, 110.0],
            "soak_min": [45.0, 75.0], "cool_rate": [60.0, 100.0, 140.0],
            "hold573_min": [0.0, 45.0]}
    total = 1
    for _v in grid.values():
        total *= len(_v)
    soak = base["soak_temp"] - (10.0 if defect_type == "色差" else 0.0) + zone_bias * 0.0
    soak = float(np.clip(soak, *BOUNDS["soak_temp"]))

    cands = []
    for r1 in grid["rate1"]:
        for r2 in grid["rate2"]:
            for sm in grid["soak_min"]:
                for cr in grid["cool_rate"]:
                    for h5 in grid["hold573_min"]:
                        p = {"rate1": r1, "rate2": r2, "soak_temp": soak,
                             "soak_min": sm, "cool_rate": cr, "hold573_min": h5}
                        m = _quick(p)
                        if m["cycle_min"] > limit_h * 60:
                            continue
                        # 缺陷针对性加权：泡/孔重升温与保温，裂纹重冷却
                        w = 1.0
                        if defect_type in ("釉泡", "针孔"):
                            w += 0.35 * (r2 / 110.0) + 0.25 * (45.0 / max(sm, 1))
                        elif defect_type == "裂纹":
                            w += 0.45 * (cr / 140.0)
                        elif defect_type == "色差":
                            w += 0.3 * (r2 / 110.0)
                        cands.append((m["total_cost"] * w, p, m))
    cands.sort(key=lambda x: x[0])
    best_p, best_m = cands[0][1], simulate(cands[0][1], with_series=True)
    base_m = simulate(base, with_series=True)
    return {
        "baseline": base_m,
        "recommended": best_m,
        "delta": {
            "risk": round(best_m["risk"] - base_m["risk"], 1),
            "energy": round(best_m["energy_kwh_m2"] - base_m["energy_kwh_m2"], 2),
            "cost": round(best_m["total_cost"] - base_m["total_cost"], 2),
            "cycle": round(best_m["cycle_min"] - base_m["cycle_min"], 1),
        },
        "top": [{"params": {k: round(float(v), 1) for k, v in p.items()},
                 "cost": round(m["total_cost"], 2), "risk": round(m["risk"], 1)}
                for _, p, m in cands[:5]],
        "search_space": len(cands),
        "search_total": total,
    }


def zone_of(u, v):
    """把缺陷在画面中的相对位置映射到窑位与温区偏差。"""
    u = float(np.clip(u, 0, 1))
    v = float(np.clip(v, 0, 1))
    layer = "上层" if v < 0.34 else ("下层" if v > 0.66 else "中层")
    area = "左火道" if u < 0.32 else ("右火道" if u > 0.68 else "中部")
    bias = (6 if layer == "上层" else (-5 if layer == "下层" else 0)) + (8 if area != "中部" else 1)
    k = 1 + int(v * 3) * 3 + int(u * 3)
    return {"zone": f"{layer}·{area}·温区{k}", "bias": bias,
            "u": round(u, 3), "v": round(v, 3)}


def defect_to_params(defect_type, zone_bias=0.0):
    """缺陷 → 工艺参数调整建议（规则来自陶瓷烧成工艺常识）。"""
    p = dict(BASE)
    if defect_type in ("釉泡", "针孔"):
        p["rate2"] = 70.0
        p["rate1"] = 85.0
        p["soak_min"] = 75.0
        p["cool_rate"] = 100.0
        p["hold573_min"] = 30.0
    elif defect_type == "裂纹":
        p["cool_rate"] = 55.0
        p["hold573_min"] = 60.0
        p["rate2"] = 80.0
    elif defect_type == "色差":
        p["soak_temp"] = 1170.0
        p["rate2"] = 70.0
        p["soak_min"] = 75.0
        p["cool_rate"] = 90.0
        p["hold573_min"] = 30.0
    else:  # 落渣：温度不是主因，只做温和优化
        p["cool_rate"] = 100.0
        p["hold573_min"] = 30.0
        p["soak_min"] = 60.0
    if zone_bias >= 10:
        p["soak_temp"] -= 5.0
    p["soak_temp"] = float(np.clip(p["soak_temp"], *BOUNDS["soak_temp"]))
    return p
