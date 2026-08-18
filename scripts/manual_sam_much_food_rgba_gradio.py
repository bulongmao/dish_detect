#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
人工交互式 SAM 食物 RGBA 生成工具（Gradio 网页版）

功能：
- 左键点击由“前景点/背景点”单选框决定点类型；
- SAM 每次返回 3 个候选，可切换候选序号；
- 食物由多个分离部分组成时，可逐块“加入最终 Mask”；
- 保存 *_food_mask.png、*_food_rgba.png、*_food_vis.jpg；
- 输出命名可直接被现有空碗拼接脚本读取。

启动：
    python manual_sam_much_food_rgba_gradio.py --device cuda:0 --port 7860

"""

from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from segment_anything import SamPredictor, sam_model_registry


# =========================================================
# 1. 路径配置
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_IMAGE_ROOT = PROJECT_ROOT / "much_food" / "images"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "much_food" / "sam"
UPLOAD_SUBDIR = "uploaded"
CHECKPOINT_MODEL_TYPES = {
    "sam_vit_b": "vit_b",
    "sam_vit_l": "vit_l",
    "sam_vit_h": "vit_h",
}


def find_default_checkpoint() -> Path:
    checkpoint_root = PROJECT_ROOT / "checkpoints"
    candidates = (
        checkpoint_root / "sam_vit_b_01ec64.pth",
        checkpoint_root / "sam_vit_l_0b3195.pth",
        checkpoint_root / "sam_vit_h_4b8939.pth",
    )
    return next((path for path in candidates if path.exists()), candidates[0])


INPUT_IMAGE_ROOT = DEFAULT_INPUT_IMAGE_ROOT
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT

MASK_ROOT = OUTPUT_ROOT / "sam_food_masks"
RGBA_ROOT = OUTPUT_ROOT / "sam_food_rgba"
VIS_ROOT = OUTPUT_ROOT / "sam_food_vis"
META_CSV = OUTPUT_ROOT / "manual_sam_food_meta.csv"

SAM_CHECKPOINT = find_default_checkpoint()
SAM_MODEL_TYPE = "vit_b"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

PREDICTOR: Optional[SamPredictor] = None
CURRENT_IMAGE_PATH: Optional[str] = None


# =========================================================
# 2. 参数与文件工具
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="人工交互式 SAM 食物 RGBA 生成工具"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_IMAGE_ROOT,
        help="待处理图片根目录，默认使用项目内 much_food/images",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Mask、RGBA、可视化和 CSV 输出根目录",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=find_default_checkpoint(),
        help="SAM 权重路径",
    )
    parser.add_argument(
        "--model-type",
        choices=sorted(sam_model_registry.keys()),
        default=None,
        help="SAM 模型类型；默认根据权重文件名自动识别",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="例如 cuda:0 或 cpu；默认优先使用 cuda:0",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅检查路径、依赖、图片和 CUDA，不加载模型或启动网页",
    )
    return parser.parse_args()


def natural_sort_key(text: str):
    parts = re.split(r"(\d+)", text.lower())
    return [int(x) if x.isdigit() else x for x in parts]


def collect_images(root: Path) -> List[Path]:
    paths = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort(key=lambda p: natural_sort_key(str(p)))
    return paths


def configure_runtime_paths(
    input_root: Path,
    output_root: Path,
    checkpoint: Path,
    model_type: Optional[str],
):
    global INPUT_IMAGE_ROOT, OUTPUT_ROOT
    global MASK_ROOT, RGBA_ROOT, VIS_ROOT, META_CSV
    global SAM_CHECKPOINT, SAM_MODEL_TYPE
    global CURRENT_IMAGE_PATH

    INPUT_IMAGE_ROOT = input_root.expanduser().resolve()
    OUTPUT_ROOT = output_root.expanduser().resolve()
    MASK_ROOT = OUTPUT_ROOT / "sam_food_masks"
    RGBA_ROOT = OUTPUT_ROOT / "sam_food_rgba"
    VIS_ROOT = OUTPUT_ROOT / "sam_food_vis"
    META_CSV = OUTPUT_ROOT / "manual_sam_food_meta.csv"
    SAM_CHECKPOINT = checkpoint.expanduser().resolve()
    SAM_MODEL_TYPE = infer_model_type(SAM_CHECKPOINT, model_type)
    CURRENT_IMAGE_PATH = None


def infer_model_type(
    checkpoint: Path,
    requested_model_type: Optional[str],
) -> str:
    if requested_model_type:
        return requested_model_type

    checkpoint_name = checkpoint.name.lower()
    for name_fragment, model_type in CHECKPOINT_MODEL_TYPES.items():
        if name_fragment in checkpoint_name:
            return model_type

    raise ValueError(
        "无法从权重文件名识别模型类型，请显式传入 "
        "--model-type vit_b、vit_l 或 vit_h"
    )


def resolve_device(requested_device: Optional[str]) -> str:
    device = requested_device or (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    torch_device = torch.device(device)

    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"请求使用 {device}，但当前 PyTorch 无法使用 CUDA"
            )
        device_index = (
            torch.cuda.current_device()
            if torch_device.index is None
            else torch_device.index
        )
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"请求使用 {device}，但只检测到 "
                f"{torch.cuda.device_count()} 张 GPU"
            )

    return str(torch_device)


def ensure_dirs():
    for p in (MASK_ROOT, RGBA_ROOT, VIS_ROOT):
        p.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def resolve_input_image(image_name: str) -> Path:
    path = (INPUT_IMAGE_ROOT / image_name).resolve()
    if not path.is_relative_to(INPUT_IMAGE_ROOT):
        raise ValueError(f"图片不在输入目录内：{image_name}")
    return path


def load_rgb(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            return np.array(normalized, dtype=np.uint8, copy=True)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"无法解析图片：{path}") from error


def resolve_uploaded_path(uploaded_file: Any) -> Path:
    if isinstance(uploaded_file, (str, Path)):
        path = Path(uploaded_file)
    elif getattr(uploaded_file, "path", None):
        path = Path(uploaded_file.path)
    elif getattr(uploaded_file, "name", None):
        path = Path(uploaded_file.name)
    else:
        raise ValueError("没有收到可解析的本地图片文件")

    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"上传的临时文件不存在：{path}")
    return path


def sanitize_upload_stem(file_name: str) -> str:
    stem = Path(file_name).stem.strip()
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem)
    stem = re.sub(r"\s+", "_", stem).strip(" ._")
    return stem[:100] or "uploaded_image"


def encode_rgb_png(image_rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb, mode="RGB").save(
        buffer,
        format="PNG",
        optimize=True,
    )
    return buffer.getvalue()


def persist_uploaded_image(uploaded_file: Any) -> Tuple[Path, np.ndarray]:
    source_path = resolve_uploaded_path(uploaded_file)
    image_rgb = load_rgb(source_path)
    png_bytes = encode_rgb_png(image_rgb)

    upload_root = INPUT_IMAGE_ROOT / UPLOAD_SUBDIR
    upload_root.mkdir(parents=True, exist_ok=True)
    stem = sanitize_upload_stem(source_path.name)

    destination = upload_root / f"{stem}.png"
    suffix = 2
    while destination.exists() and destination.read_bytes() != png_bytes:
        destination = upload_root / f"{stem}_{suffix}.png"
        suffix += 1

    if not destination.exists():
        destination.write_bytes(png_bytes)

    return destination.resolve(), image_rgb


def empty_state() -> Dict[str, Any]:
    return {
        "image_name": "",
        "image_path": "",
        "points": [],
        "labels": [],
        "masks": None,
        "scores": None,
        "candidate_index": 0,
        "committed_mask": None,
    }


def set_predictor_image(image_path: str, image_rgb: np.ndarray):
    global CURRENT_IMAGE_PATH
    if PREDICTOR is None:
        raise RuntimeError("SAM Predictor 未初始化")
    if CURRENT_IMAGE_PATH != image_path:
        PREDICTOR.set_image(image_rgb)
        CURRENT_IMAGE_PATH = image_path


# =========================================================
# 3. Mask 后处理
# =========================================================

def fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    binary = np.where(mask_u8 > 0, 255, 0).astype(np.uint8)
    h, w = binary.shape
    padded = cv2.copyMakeBorder(
        binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
    )
    flood = padded.copy()
    flood_mask = np.zeros(
        (flood.shape[0] + 2, flood.shape[1] + 2),
        dtype=np.uint8,
    )
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood = flood[1:h + 1, 1:w + 1]
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, holes)


def postprocess_mask(
    mask_u8: np.ndarray,
    use_fill_holes: bool,
    close_kernel: int,
    erode_iterations: int,
) -> np.ndarray:
    result = np.where(mask_u8 > 0, 255, 0).astype(np.uint8)

    if use_fill_holes:
        result = fill_holes(result)

    close_kernel = int(close_kernel)
    if close_kernel >= 3:
        if close_kernel % 2 == 0:
            close_kernel += 1
        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        result = cv2.morphologyEx(
            result, cv2.MORPH_CLOSE, kernel, iterations=1
        )

    erode_iterations = int(erode_iterations)
    if erode_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        result = cv2.erode(
            result, kernel, iterations=erode_iterations
        )

    return result


# =========================================================
# 4. SAM 与显示
# =========================================================

def predict_state(state: Dict[str, Any], image_rgb: np.ndarray):
    if not state.get("image_path") or not state["points"]:
        state["masks"] = None
        state["scores"] = None
        state["candidate_index"] = 0
        return state

    set_predictor_image(state["image_path"], image_rgb)

    with torch.inference_mode():
        masks, scores, _ = PREDICTOR.predict(
            point_coords=np.asarray(state["points"], dtype=np.float32),
            point_labels=np.asarray(state["labels"], dtype=np.int32),
            box=None,
            multimask_output=True,
        )

    order = np.argsort(scores)[::-1]
    state["masks"] = masks[order]
    state["scores"] = scores[order]
    state["candidate_index"] = 0
    return state


def current_mask(state: Dict[str, Any]) -> Optional[np.ndarray]:
    masks = state.get("masks")
    if masks is None or len(masks) == 0:
        return None
    index = int(np.clip(state.get("candidate_index", 0), 0, len(masks) - 1))
    return masks[index].astype(bool)


def final_mask(state: Dict[str, Any]) -> Optional[np.ndarray]:
    committed = state.get("committed_mask")
    current = current_mask(state)

    if committed is None and current is None:
        return None
    if committed is None:
        return current.copy()
    if current is None:
        return committed.copy()
    return np.logical_or(committed, current)


def overlay_image(
    image_rgb: np.ndarray,
    state: Dict[str, Any],
    use_fill_holes: bool = False,
    close_kernel: int = 0,
    erode_iterations: int = 0,
) -> np.ndarray:
    view = image_rgb.copy()

    committed = state.get("committed_mask")
    current = current_mask(state)

    if committed is not None:
        mask = committed.astype(bool)
        green = np.zeros_like(view)
        green[:, :, 1] = 255
        view[mask] = (view[mask] * 0.70 + green[mask] * 0.30).astype(np.uint8)

    if current is not None:
        mask = current.astype(bool)
        red = np.zeros_like(view)
        red[:, :, 0] = 255
        view[mask] = (view[mask] * 0.62 + red[mask] * 0.38).astype(np.uint8)

    merged = final_mask(state)
    if merged is not None and (
        use_fill_holes or int(close_kernel) >= 3 or int(erode_iterations) > 0
    ):
        processed = postprocess_mask(
            merged.astype(np.uint8) * 255,
            use_fill_holes,
            int(close_kernel),
            int(erode_iterations),
        )
        contours, _ = cv2.findContours(
            processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        cv2.drawContours(bgr, contours, -1, (0, 255, 255), 2)
        view = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
    for (x, y), label in zip(state["points"], state["labels"]):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(bgr, (int(x), int(y)), 7, color, -1)
        cv2.circle(bgr, (int(x), int(y)), 9, (255, 255, 255), 1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def status_text(state: Dict[str, Any]) -> str:
    scores = state.get("scores")
    score_text = ""
    if scores is not None and len(scores) > 0:
        idx = int(np.clip(state.get("candidate_index", 0), 0, len(scores) - 1))
        score_text = f"；当前候选分数={float(scores[idx]):.4f}"

    committed = state.get("committed_mask")
    committed_area = int(np.count_nonzero(committed)) if committed is not None else 0

    return (
        f"前景点={state['labels'].count(1)}；"
        f"背景点={state['labels'].count(0)}；"
        f"已加入最终 Mask 像素={committed_area}"
        f"{score_text}"
    )


# =========================================================
# 5. Gradio 回调
# =========================================================

def load_selected(image_name: str):
    if not image_name:
        return empty_state(), None, "请选择图片", 0

    try:
        path = resolve_input_image(image_name)
    except ValueError as error:
        return empty_state(), None, str(error), 0

    if not path.exists():
        return empty_state(), None, f"图片不存在：{path}", 0

    try:
        image = load_rgb(path)
    except ValueError as error:
        return empty_state(), None, str(error), 0

    state = empty_state()
    state["image_name"] = image_name
    state["image_path"] = str(path)
    set_predictor_image(str(path), image)
    return state, image, f"已加载：{image_name}", 0


def load_uploaded(uploaded_file: Any):
    if not uploaded_file:
        return empty_state(), None, "请选择本地图片", 0, None

    try:
        path, image = persist_uploaded_image(uploaded_file)
        image_name = str(path.relative_to(INPUT_IMAGE_ROOT))
        state = empty_state()
        state["image_name"] = image_name
        state["image_path"] = str(path)
        set_predictor_image(str(path), image)
    except (OSError, ValueError) as error:
        return empty_state(), None, str(error), 0, None

    return (
        state,
        image,
        "本地图片已上传并加载\n"
        f"原图：{path}\n"
        f"分割结果将保存到：{OUTPUT_ROOT}",
        0,
        None,
    )


def add_click(
    state: Dict[str, Any],
    label_mode: str,
    use_fill_holes: bool,
    close_kernel: int,
    erode_iterations: int,
    evt: gr.SelectData,
):
    if not state or not state.get("image_path"):
        return state, None, "请先选择图片", 0

    image = load_rgb(Path(state["image_path"]))
    if evt.index is None:
        return state, image, status_text(state), state.get("candidate_index", 0)

    x, y = evt.index
    x = int(np.clip(x, 0, image.shape[1] - 1))
    y = int(np.clip(y, 0, image.shape[0] - 1))
    label = 1 if label_mode == "前景点" else 0

    state["points"].append([x, y])
    state["labels"].append(label)
    state = predict_state(state, image)

    return (
        state,
        overlay_image(image, state, use_fill_holes, close_kernel, erode_iterations),
        status_text(state),
        int(state.get("candidate_index", 0)),
    )


def choose_candidate(
    state: Dict[str, Any],
    candidate_index: int,
    use_fill_holes: bool,
    close_kernel: int,
    erode_iterations: int,
):
    if not state or not state.get("image_path"):
        return state, None, "请先选择图片"

    masks = state.get("masks")
    if masks is not None and len(masks) > 0:
        state["candidate_index"] = int(np.clip(candidate_index, 0, len(masks) - 1))

    image = load_rgb(Path(state["image_path"]))
    return (
        state,
        overlay_image(image, state, use_fill_holes, close_kernel, erode_iterations),
        status_text(state),
    )


def undo_point(
    state: Dict[str, Any],
    use_fill_holes: bool,
    close_kernel: int,
    erode_iterations: int,
):
    if not state or not state.get("image_path"):
        return state, None, "请先选择图片", 0

    if state["points"]:
        state["points"].pop()
        state["labels"].pop()

    image = load_rgb(Path(state["image_path"]))
    state = predict_state(state, image)
    return (
        state,
        overlay_image(image, state, use_fill_holes, close_kernel, erode_iterations),
        status_text(state),
        int(state.get("candidate_index", 0)),
    )


def clear_current(
    state: Dict[str, Any],
    use_fill_holes: bool,
    close_kernel: int,
    erode_iterations: int,
):
    if not state or not state.get("image_path"):
        return state, None, "请先选择图片", 0

    state["points"] = []
    state["labels"] = []
    state["masks"] = None
    state["scores"] = None
    state["candidate_index"] = 0

    image = load_rgb(Path(state["image_path"]))
    return (
        state,
        overlay_image(image, state, use_fill_holes, close_kernel, erode_iterations),
        status_text(state),
        0,
    )


def commit_mask(
    state: Dict[str, Any],
    use_fill_holes: bool,
    close_kernel: int,
    erode_iterations: int,
):
    if not state or not state.get("image_path"):
        return state, None, "请先选择图片", 0

    mask = current_mask(state)
    image = load_rgb(Path(state["image_path"]))

    if mask is None:
        return (
            state,
            overlay_image(image, state, use_fill_holes, close_kernel, erode_iterations),
            "当前没有可加入的 SAM Mask",
            int(state.get("candidate_index", 0)),
        )

    if state["committed_mask"] is None:
        state["committed_mask"] = mask.copy()
    else:
        state["committed_mask"] = np.logical_or(state["committed_mask"], mask)

    state["points"] = []
    state["labels"] = []
    state["masks"] = None
    state["scores"] = None
    state["candidate_index"] = 0

    return (
        state,
        overlay_image(image, state, use_fill_holes, close_kernel, erode_iterations),
        "已加入最终 Mask；可继续点击下一块食物",
        0,
    )


def reset_selection(state: Dict[str, Any]):
    if not state or not state.get("image_path"):
        return empty_state(), None, "请先选择图片", 0

    new_state = empty_state()
    new_state["image_name"] = state["image_name"]
    new_state["image_path"] = state["image_path"]
    image = load_rgb(Path(state["image_path"]))
    set_predictor_image(state["image_path"], image)
    return new_state, image, "已重置全部选择", 0


def refresh_preview(
    state: Dict[str, Any],
    use_fill_holes: bool,
    close_kernel: int,
    erode_iterations: int,
):
    if not state or not state.get("image_path"):
        return None, "请先选择图片"
    image = load_rgb(Path(state["image_path"]))
    return (
        overlay_image(image, state, use_fill_holes, close_kernel, erode_iterations),
        status_text(state),
    )


def append_meta(
    source: Path,
    mask_path: Path,
    rgba_path: Path,
    vis_path: Path,
    area: int,
):
    exists = META_CSV.exists()
    with META_CSV.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow([
                "source_image", "food_mask", "food_rgba",
                "visualization", "mask_area"
            ])
        writer.writerow([
            str(source), str(mask_path), str(rgba_path),
            str(vis_path), area
        ])


def save_result(
    state: Dict[str, Any],
    use_fill_holes: bool,
    close_kernel: int,
    erode_iterations: int,
):
    if not state or not state.get("image_path"):
        return None, "请先选择图片"

    merged = final_mask(state)
    if merged is None:
        return None, "当前没有 Mask，无法保存"

    source_path = Path(state["image_path"])
    image = load_rgb(source_path)
    mask_u8 = postprocess_mask(
        merged.astype(np.uint8) * 255,
        use_fill_holes,
        int(close_kernel),
        int(erode_iterations),
    )

    if np.count_nonzero(mask_u8) == 0:
        return None, "后处理后 Mask 为空，未保存"

    try:
        relative = source_path.resolve().relative_to(INPUT_IMAGE_ROOT)
    except ValueError:
        return None, (
            "源图片不在输入目录内，无法确定输出子目录。"
            "请通过网页的“上传本地图片”重新加载。"
        )
    relative_dir = relative.parent
    stem = source_path.stem

    mask_dir = MASK_ROOT / relative_dir
    rgba_dir = RGBA_ROOT / relative_dir
    vis_dir = VIS_ROOT / relative_dir
    for directory in (mask_dir, rgba_dir, vis_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mask_path = mask_dir / f"{stem}_food_mask.png"
    rgba_path = rgba_dir / f"{stem}_food_rgba.png"
    vis_path = vis_dir / f"{stem}_food_vis.jpg"

    rgba = np.dstack([image, mask_u8]).astype(np.uint8)
    rgba[mask_u8 == 0, :3] = 0

    Image.fromarray(mask_u8, mode="L").save(mask_path)
    Image.fromarray(rgba, mode="RGBA").save(rgba_path)

    vis = image.copy()
    mask_bool = mask_u8 > 0
    red = np.zeros_like(vis)
    red[:, :, 0] = 255
    vis[mask_bool] = (
        vis[mask_bool] * 0.62 + red[mask_bool] * 0.38
    ).astype(np.uint8)

    bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(bgr, contours, -1, (0, 255, 255), 2)
    cv2.imwrite(str(vis_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 94])

    append_meta(
        source_path, mask_path, rgba_path, vis_path,
        int(np.count_nonzero(mask_u8))
    )

    preview = np.zeros_like(image)
    preview[mask_bool] = image[mask_bool]

    return (
        preview,
        "保存成功\n"
        f"Mask：{mask_path}\n"
        f"RGBA：{rgba_path}\n"
        f"可视化：{vis_path}",
    )


# =========================================================
# 6. 界面
# =========================================================

def build_app(image_names: List[str]):
    with gr.Blocks(title="人工 SAM 食物 RGBA 工具") as app:
        state = gr.State(value=empty_state())

        gr.Markdown(
            """
# 人工 SAM 食物 RGBA 工具

1. 从项目下拉框选择图片，或使用**上传本地图片**；
2. 使用**前景点**点击食物，使用**背景点**点击盘子或误选区域；
3. 候选序号 0/1/2 可切换 SAM 输出；
4. 食物由多块组成时，逐块点击并选择 **加入最终 Mask**；
5. 保存后生成 `*_food_rgba.png`，可直接注入现有空碗拼接脚本。

显示颜色：红色为当前候选，绿色为已经加入最终 Mask 的区域，黄色轮廓为保存时后处理结果。

网页上传的原图会保存到 `much_food/images/uploaded`，对应 Mask、RGBA 和可视化会保存到输出目录下的 `uploaded` 子目录。
"""
        )

        with gr.Row():
            image_dropdown = gr.Dropdown(
                choices=image_names,
                value=image_names[0] if image_names else None,
                label="原始图片",
                interactive=True,
            )
            load_button = gr.Button("加载图片", variant="primary")
            uploaded_file = gr.File(
                label="上传本地图片",
                file_types=["image"],
                type="filepath",
                interactive=True,
            )

        with gr.Row():
            label_mode = gr.Radio(
                ["前景点", "背景点"], value="前景点",
                label="点击类型"
            )
            candidate_index = gr.Slider(
                0, 2, value=0, step=1, label="SAM 候选序号"
            )

        with gr.Row():
            use_fill_holes = gr.Checkbox(
                value=True, label="保存时填内部孔洞"
            )
            close_kernel = gr.Slider(
                0, 15, value=3, step=1,
                label="闭运算核（建议 3 或 5；0~2 为关闭）"
            )
            erode_iterations = gr.Slider(
                0, 3, value=1, step=1,
                label="向内腐蚀次数（去盘子边缘，建议 0~1）"
            )

        with gr.Row():
            image_view = gr.Image(
                label="点击图像添加提示点（也可直接拖入本地图片）",
                type="filepath",
                sources=["upload"],
                interactive=True,
            )
            rgba_preview = gr.Image(
                label="保存后的食物预览",
                type="numpy",
                interactive=False,
            )

        with gr.Row():
            undo_button = gr.Button("撤销最后一点")
            clear_button = gr.Button("清空当前提示")
            commit_button = gr.Button("加入最终 Mask", variant="secondary")
            reset_button = gr.Button("重置全部")
            preview_button = gr.Button("刷新后处理预览")
            save_button = gr.Button("保存 Mask + RGBA", variant="primary")

        status = gr.Textbox(label="状态", lines=4, interactive=False)

        load_button.click(
            load_selected, [image_dropdown],
            [state, image_view, status, candidate_index]
        )
        image_dropdown.change(
            load_selected, [image_dropdown],
            [state, image_view, status, candidate_index]
        )
        uploaded_file.upload(
            load_uploaded,
            [uploaded_file],
            [state, image_view, status, candidate_index, image_dropdown],
        )
        image_view.upload(
            load_uploaded,
            [image_view],
            [state, image_view, status, candidate_index, image_dropdown],
        )
        image_view.select(
            add_click,
            [state, label_mode, use_fill_holes, close_kernel, erode_iterations],
            [state, image_view, status, candidate_index],
        )
        candidate_index.input(
            choose_candidate,
            [state, candidate_index, use_fill_holes, close_kernel, erode_iterations],
            [state, image_view, status],
        )
        undo_button.click(
            undo_point,
            [state, use_fill_holes, close_kernel, erode_iterations],
            [state, image_view, status, candidate_index],
        )
        clear_button.click(
            clear_current,
            [state, use_fill_holes, close_kernel, erode_iterations],
            [state, image_view, status, candidate_index],
        )
        commit_button.click(
            commit_mask,
            [state, use_fill_holes, close_kernel, erode_iterations],
            [state, image_view, status, candidate_index],
        )
        reset_button.click(
            reset_selection, [state],
            [state, image_view, status, candidate_index]
        )
        preview_button.click(
            refresh_preview,
            [state, use_fill_holes, close_kernel, erode_iterations],
            [image_view, status],
        )
        save_button.click(
            save_result,
            [state, use_fill_holes, close_kernel, erode_iterations],
            [rgba_preview, status],
        )

        if image_names:
            app.load(
                load_selected, [image_dropdown],
                [state, image_view, status, candidate_index]
            )

    return app


# =========================================================
# 7. 主函数
# =========================================================

def main():
    global PREDICTOR

    args = parse_args()
    configure_runtime_paths(
        args.input_root,
        args.output_root,
        args.checkpoint,
        args.model_type,
    )

    INPUT_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(f"SAM 权重不存在：{SAM_CHECKPOINT}")

    device = resolve_device(args.device)

    image_paths = collect_images(INPUT_IMAGE_ROOT)
    image_names = [
        str(path.relative_to(INPUT_IMAGE_ROOT))
        for path in image_paths
    ]

    print("=" * 50)
    print("Manual SAM food RGBA tool")
    print("Input:", INPUT_IMAGE_ROOT)
    print("Output:", OUTPUT_ROOT)
    print("Checkpoint:", SAM_CHECKPOINT)
    print("Model type:", SAM_MODEL_TYPE)
    print("Device:", device)
    if device.startswith("cuda"):
        print("GPU:", torch.cuda.get_device_name(torch.device(device).index or 0))
    print("Images:", len(image_paths))
    print("URL: http://%s:%d" % (args.host, args.port))
    print("=" * 50)

    if args.check_only:
        print("检查通过：依赖、路径、图片和计算设备均可用。")
        return

    ensure_dirs()
    sam = sam_model_registry[SAM_MODEL_TYPE](
        checkpoint=str(SAM_CHECKPOINT)
    )
    sam.to(device=device)
    sam.eval()
    PREDICTOR = SamPredictor(sam)

    app = build_app(image_names)
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.open_browser,
        show_error=True,
    )


if __name__ == "__main__":
    main()
