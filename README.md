# Dish Detect

面向餐盘与食物图像的分割、抠图和数据增强工具集。项目以 Segment Anything Model（SAM）为核心，提供 Gradio 和 FastAPI + Web 两套交互界面，可生成食物 Mask、透明 RGBA 素材和可视化结果。

## 主要功能

- 前景点/背景点交互式 SAM 分割
- 多候选 Mask 切换和多区域合并
- 输出 `*_food_mask.png`、`*_food_rgba.png` 和 `*_food_vis.jpg`
- 餐盘裁剪、食物贴图与合成数据生成
- Gradio 快速界面与 FastAPI Web 界面

## 项目背景

餐具价格识别中，碗盘类别容易与某些菜品、颜色和摆放方式强绑定，模型因此可能学到食物纹理而不是餐具特征。本项目通过“同盘异食”合成数据打破这种伪相关：先从源图提取食物 RGBA，再将其注入目标空碗/空盘，尽量保留餐具材质、盘沿和现场背景。

## 数据增强流水线

```text
YOLO 标签裁剪餐碗 Crop
        ↓
SAM 提取 Food Mask / RGBA
        ↓
空碗检测与可填充区域构建
        ↓
缩放 + 位置搜索 + 覆盖率筛选
        ↓
合成图 + CSV 元数据 + 人工抽检
```

工程中实验了四类组合路径：

- **A1：单餐碗替换**：在单个有食物餐碗的 Crop 中替换食物。
- **A2：整图多餐碗替换**：基于检测框/ROI 逐个替换整张图中的餐碗。
- **B1：空餐碗检测**：SAM 候选 Mask 结合 NMS 和面积等规则过滤。
- **B2：inner mask 填充**：从 outer mask 构建内层缺口，以 hole coverage 为主目标搜索食物缩放与位置。

## 质量控制

- SAM 不盲目选最高分候选；结合面积、中心重叠、贴边程度和稳定性选主 Mask，再合并邻近食物块。
- 通过闭运算、填洞和连通性处理构建完整食物团块，但不用大核形态学“猜”未进入候选池的区域。
- 以 `hole coverage` 和源食物可见比例作为保存门槛，替代“看起来差不多”的主观判断。
- 使用二值 Alpha 或仅向内羽化，避免外向高斯模糊形成固定半透明光圈。
- 保留真实数据主导训练，对合成图进行人工抽检，避免模型转而学到合成伪影。

## 当前资产快照

| 项目 | 数量 |
|---|---:|
| 当前处理的食物输入图 | 2,048 |
| 目标素材食物 RGBA | 97 |
| 单盘 + 多盘合成 JPG | 2,560 |

> 上述数字来自当前项目汇报快照，反映阶段性资产规模，不等同于最终训练集或效果指标。

## 目录

```text
scripts/             分割、裁剪、贴图和启动脚本
web/manual_sam/      Web 前端资源
much_food/           默认输入/输出工作目录
checkpoints/         SAM 权重目录（未入库）
```

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-manual-sam.txt
```

将 SAM 权重放入 `checkpoints/`，例如 `checkpoints/sam_vit_b_01ec64.pth`。

```bash
# Gradio 界面
python scripts/manual_sam_much_food_rgba_gradio.py --device cuda:0 --port 7860 --open-browser

# FastAPI + Web 界面
python scripts/manual_sam_much_food_rgba_web.py --device cuda:0 --port 8000 --open-browser

# 仅检查环境与路径
python scripts/manual_sam_much_food_rgba_gradio.py --check-only
```

默认输入为 `much_food/images/`，输出为 `much_food/sam/`。可用 `--input-root`、`--output-root`、`--checkpoint` 和 `--model-type` 覆盖。

## 注意

- SAM 权重、原始图片和生成结果不纳入 Git。
- `requirements.txt` 是完整开发环境快照，含特定 CUDA 版本和本地路径；仅运行人工 SAM 工具时优先使用 `requirements-manual-sam.txt`。
- CPU 可运行，但速度会明显慢于 CUDA GPU。
