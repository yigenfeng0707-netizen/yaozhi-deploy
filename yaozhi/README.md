# 窑智 · 陶瓷工艺 AI 大脑（天元杯参赛作品）

离线可运行的陶瓷釉面质检 + 窑温寻优 + 纹样生成一体化服务（CPU 推理，无需联网）。

## 快速开始

```
pip install -r requirements.txt
python tools/bootstrap.py        # 一键自检：生成仿真样本 → 建特征库 → 输出评估报告（约 3~5 分钟）
run.bat                          # 或 python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

浏览器访问 http://127.0.0.1:8000

## 离线说明

- ResNet18 预训练权重已随包携带（`models/torch_hub/hub/checkpoints/resnet18-f37072fd.pth`，44.6MB），首次运行无需联网下载。
- 权重目录由 `core/defect.py` 在导入时以 `TORCH_HOME` 指向包内目录，亦可自行用环境变量覆盖。
- 服务器无 GPU 依赖，全部能力在本机 CPU 上完成推理。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `app.py` | FastAPI 服务与全部 REST API |
| `core/defect.py` | 缺陷检测内核：PatchCore 轻量版（ResNet18 中层特征 + coreset 记忆库 + Lab 色差通道） |
| `core/kiln.py` | 窑温数字孪生（一维瞬态热传导隐式差分）与曲线寻优 |
| `core/pattern.py` | 参数化纹样生成与可制造性校验 |
| `core/synth.py` | 仿真釉面样本生成（自检与演示用） |
| `tools/bootstrap.py` | 一键自检：生成数据集、重建特征库、评估与寻优报告 |
| `static/` | 前端界面 |
| `models/` | 特征库 `bank.npz`、自检报告 `eval_report.json`、离线权重 |

## API 摘要

- `GET /api/health`、`GET /api/status` — 健康与状态
- `POST /api/detect` — 上传釉面图，返回缺陷判定 / 定位框 / 热力图 / 处置建议
- `POST /api/calibrate` — 现场校准：上传 ≥6 张（建议 20~40 张）同釉色族良品图重建特征库；覆盖前自动备份为 `bank.prev.npz`
- `GET /api/kiln/simulate`、`POST /api/kiln/optimize` — 窑温曲线仿真与成本-风险寻优
- `GET /api/pattern/generate`、`GET /api/pattern/options` — 纹样生成与可制造性校验
- `POST /api/demo` — 一键闭环演示：缺陷样本 → 检测定位 → 窑位/温区 → 参数寻优 → 下一窑首件复检（界面明确标注仿真数据）

## 自检结果

`python tools/bootstrap.py` 生成 `models/eval_report.json`，包含：缺陷检出率、良品误报、温度曲线物理校验、窑温寻优前后对比与纹样校验明细。
