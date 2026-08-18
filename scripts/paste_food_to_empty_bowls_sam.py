#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用本地 SAM，把已有 food_rgba 图片拼接到空碗图片中。

当前适用：
- 每张空碗图片中只有一个碗，位置不固定；
- 银色拍摄背景；
- 已有 food_rgba（4 通道 PNG，Alpha 为食物 Mask）。

后续扩展：
- 将 EXPECTED_CONTAINERS_PER_IMAGE 改为 0，可保留同图中的多个碗；
- 每个检测到的碗都会独立填充食物。

运行示例：
    python paste_food_to_empty_bowls_sam.py --device cuda:0 --max_images 20

四卡运行：
    python paste_food_to_empty_bowls_sam.py --launch_4gpu --gpu_ids 0,1,2,3
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# =========================================================
# 1. 路径配置
# =========================================================

# 新拍摄的空碗整图
EMPTY_IMAGE_ROOT = Path(
    "/data/ljy/dish_detect/many_plate_empty/plate_image"
)

# 已有食物 RGBA
FOOD_RGBA_ROOT = Path(
    "/data/ljy/dish_detect/much_food/sam/sam_food_rgba"
)

# 输出
OUTPUT_ROOT = Path(
    "/data/ljy/dish_detect/many_empty_plate_filled"
)

OUTPUT_IMAGE_ROOT = OUTPUT_ROOT / "images"
OUTER_MASK_ROOT = OUTPUT_ROOT / "outer_masks"
INNER_MASK_ROOT = OUTPUT_ROOT / "inner_masks"
VIS_ROOT = OUTPUT_ROOT / "visualizations"
META_ROOT = OUTPUT_ROOT / "meta"
LOG_ROOT = OUTPUT_ROOT / "logs_4gpu"

# SAM 权重
SAM_CHECKPOINT = Path(
    "/data/ljy/dish_detect/checkpoints/sam_vit_h_4b8939.pth"
)

SAM_MODEL_TYPE = "vit_h"


# =========================================================
# 2. 运行与输出配置
# =========================================================

# 每张空碗图生成多少个不同食物版本
NUM_VARIANTS_PER_IMAGE = 5

# 当前每张图只有一个碗，设为 1。
# 未来一张图多个碗时改为 0，表示保留所有有效实例。
EXPECTED_CONTAINERS_PER_IMAGE = 0

MAX_CONTAINERS_PER_IMAGE = 6

# 多碗时，是否要求所有碗都成功填入食物
REQUIRE_ALL_CONTAINERS_FILLED = True

# 同一张合成图中是否禁止重复使用同一个 food_rgba
UNIQUE_FOOD_WITHIN_IMAGE = True

SKIP_EXISTING_OUTPUT = True
JPEG_QUALITY = 95
RANDOM_SEED = 42


# =========================================================
# 3. SAM 自动 Mask 参数
# =========================================================

SAM_POINTS_PER_SIDE = 24
SAM_PRED_IOU_THRESH = 0.86
SAM_STABILITY_SCORE_THRESH = 0.90

# 候选过碎可改为 0
SAM_CROP_N_LAYERS = 1
SAM_CROP_N_POINTS_DOWNSCALE_FACTOR = 2
SAM_MIN_MASK_REGION_AREA = 300


# =========================================================
# 4. 餐具候选筛选参数
# =========================================================

MIN_CONTAINER_AREA_RATIO = 0.015
MAX_CONTAINER_AREA_RATIO = 0.65
MAX_CONTAINER_BBOX_AREA_RATIO = 0.85

MIN_CONTAINER_ASPECT_RATIO = 0.35
MAX_CONTAINER_ASPECT_RATIO = 3.20

MAX_BORDER_TOUCH_RATIO = 0.08
MIN_LARGEST_COMPONENT_RATIO = 0.82
MIN_CONTAINER_SOLIDITY = 0.50
MIN_CONTAINER_COMPACTNESS = 0.30

CONTAINER_NMS_IOU_THRESHOLD = 0.55
CONTAINER_CONTAINMENT_THRESHOLD = 0.85


# =========================================================
# 5. 餐具内部区域参数
# =========================================================

# 越大，inner_mask 越小，越不容易覆盖碗沿
INNER_DISTANCE_RATIO = 0.17

INNER_MIN_SHRINK_PIXELS = 3
MIN_INNER_TO_OUTER_RATIO = 0.28
MAX_INNER_TO_OUTER_RATIO = 0.82

OUTER_CLOSE_KERNEL = 5
OUTER_CLOSE_ITERATIONS = 1


# =========================================================
# 6. 右侧干扰区域开关
# =========================================================

# True：禁止把右侧区域识别为餐具，也禁止食物显示到该区域
# False：完全关闭右侧屏蔽
ENABLE_RIGHT_EXCLUSION = True

# 从整张图宽度的多少位置开始屏蔽右侧。
# 示例图中裤腿大约从 x=0.76W 开始，因此默认设为 0.76。
RIGHT_EXCLUSION_START_RATIO = 0.76

# 可选：直接指定绝对像素横坐标。
# None 表示使用 RIGHT_EXCLUSION_START_RATIO；
# 例如设为 980，则从 x=980 到图片最右侧全部禁用。
RIGHT_EXCLUSION_START_X = None

# 一个 SAM 候选 Mask 有多少比例进入右侧禁区后，直接拒绝为餐具。
# 0.02 表示候选有超过 2% 位于禁区，就不把它当作碗。
MAX_CONTAINER_RIGHT_OVERLAP_RATIO = 0.02

# 可视化中是否把右侧禁区涂成红色半透明区域
DRAW_RIGHT_EXCLUSION_IN_VIS = True


# =========================================================
# 6. 食物放置参数
# =========================================================

SOURCE_ALPHA_THRESHOLD = 128

# 食物基础尺寸相对 inner_mask 的比例
FOOD_BASE_FILL_RATIO = 1.02

FOOD_SCALE_MULTIPLIERS = (
    0.95,
    1.05,
    1.15,
    1.25,
    1.35,
)

FOOD_OFFSET_RATIOS = (
    0.00,
    0.025,
    0.05,
)

# 食物前景至少有多少比例落在碗内
MIN_FOOD_INSIDE_RATIO = 0.84

# 希望食物覆盖 inner_mask 的大致比例
TARGET_INNER_COVERAGE = 0.72

MAX_FOOD_UPSCALE = 3.5
MAX_SOURCE_TRIALS = 300

# 二值 Alpha 可减少半透明外围
USE_BINARY_SOURCE_ALPHA = True

# 只向内羽化，不向外扩散
INWARD_FEATHER_PIXELS = 2.0


# =========================================================
# 6.1 食物与餐盘形状匹配
# =========================================================

# 总开关：True 启用长方形食物与长方形餐盘匹配
ENABLE_SHAPE_MATCHING = True

# 长方形/长条形食物只能放进长方形餐盘
RECTANGULAR_FOOD_ONLY_TO_RECTANGULAR_CONTAINER = True

# 自动旋转长方形食物，使食物长边与餐盘长边平行
ALIGN_RECTANGULAR_LONG_AXIS = True

# 食物的长短边比例达到该值，才可能视为长方形/长条形
FOOD_RECT_ASPECT_THRESHOLD = 1.45

# 食物前景面积 / 最小外接旋转矩形面积
FOOD_RECTANGULARITY_THRESHOLD = 0.50

# 餐盘 inner_mask 的长短边比例达到该值，才可能视为长方形餐盘
CONTAINER_RECT_ASPECT_THRESHOLD = 1.35

# 餐盘 inner_mask 面积 / 最小外接旋转矩形面积
# 理想椭圆约为 0.785；长方形通常更接近 1
CONTAINER_RECTANGULARITY_THRESHOLD = 0.80

# 是否限制食物与餐盘的长短边比例差异
ENABLE_RECT_ASPECT_RATIO_MATCH = True
MAX_RECT_ASPECT_RATIO_FACTOR = 1.80

# 计算形状时至少需要的前景像素数
MIN_SHAPE_MASK_PIXELS = 100

# 旋转后长轴允许的最大误差（度）
MAX_LONG_AXIS_ALIGNMENT_ERROR = 8.0

# 旋转后重新测量并验证长轴方向
VERIFY_LONG_AXIS_AFTER_ROTATION = True


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# =========================================================
# 7. 数据结构
# =========================================================

@dataclass
class ContainerInstance:
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    score: float
    area_ratio: float
    predicted_iou: float
    stability_score: float
    solidity: float
    compactness: float


# =========================================================
# 8. 参数解析
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="使用本地 SAM 将 food_rgba 放入空碗"
    )

    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_images", type=int, default=0)

    parser.add_argument(
        "--launch_4gpu",
        action="store_true",
    )

    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0,1,2,3",
    )

    return parser.parse_args()


# =========================================================
# 9. 通用函数
# =========================================================

def natural_sort_key(text: str):
    parts = re.split(r"(\d+)", text.lower())
    return [
        int(part) if part.isdigit() else part
        for part in parts
    ]


def collect_images(root: Path) -> List[Path]:
    if not root.exists():
        return []

    paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    paths.sort(
        key=lambda path: natural_sort_key(str(path))
    )

    return paths


def collect_food_rgba(root: Path) -> List[Path]:
    if not root.exists():
        return []

    paths = [
        path
        for path in root.rglob("*.png")
        if path.is_file()
        and (
            path.name.endswith("_food_rgba.png")
            or path.name.endswith("_rgba.png")
        )
    ]

    paths.sort(
        key=lambda path: natural_sort_key(str(path))
    )

    return paths


def ensure_output_dirs():
    for directory in (
        OUTPUT_IMAGE_ROOT,
        OUTER_MASK_ROOT,
        INNER_MASK_ROOT,
        VIS_ROOT,
        META_ROOT,
        LOG_ROOT,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def validate_rank(rank: int, world_size: int):
    if world_size <= 0:
        raise ValueError("world_size 必须大于 0")

    if rank < 0 or rank >= world_size:
        raise ValueError(
            "rank 必须满足 0 <= rank < world_size"
        )


def resolve_device(
    requested_device: Optional[str],
    rank: int,
) -> str:
    if requested_device:
        return requested_device

    import torch

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        return f"cuda:{rank % max(gpu_count, 1)}"

    return "cpu"


def safe_relative_stem(image_path: Path) -> str:
    relative = image_path.relative_to(
        EMPTY_IMAGE_ROOT
    ).with_suffix("")

    text = "__".join(relative.parts)

    return re.sub(
        r"[^0-9a-zA-Z_\-\u4e00-\u9fff]",
        "_",
        text,
    )


def get_right_exclusion_start_x(
    image_width: int,
) -> int:
    """计算右侧禁区起点。"""
    if RIGHT_EXCLUSION_START_X is not None:
        start_x = int(RIGHT_EXCLUSION_START_X)
    else:
        start_x = int(round(
            image_width * RIGHT_EXCLUSION_START_RATIO
        ))

    return int(np.clip(start_x, 0, image_width))


def build_right_exclusion_mask(
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """
    返回二值禁区 Mask：
    - 255：禁止识别餐具、禁止显示食物
    - 0：允许区域
    """
    mask = np.zeros(
        (image_height, image_width),
        dtype=np.uint8,
    )

    if not ENABLE_RIGHT_EXCLUSION:
        return mask

    start_x = get_right_exclusion_start_x(image_width)
    mask[:, start_x:] = 255
    return mask


def build_allowed_region_mask(
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """
    返回允许区域 Mask：
    - 255：允许餐具和食物
    - 0：右侧禁区
    """
    exclusion = build_right_exclusion_mask(
        image_height,
        image_width,
    )
    return cv2.bitwise_not(exclusion)


def calculate_right_exclusion_overlap(
    mask_u8: np.ndarray,
) -> float:
    """计算候选 Mask 中有多少比例落入右侧禁区。"""
    if not ENABLE_RIGHT_EXCLUSION:
        return 0.0

    height, width = mask_u8.shape[:2]
    exclusion = build_right_exclusion_mask(height, width)

    mask_bool = mask_u8 > 0
    mask_area = int(np.count_nonzero(mask_bool))
    if mask_area == 0:
        return 0.0

    overlap_area = int(np.count_nonzero(
        np.logical_and(mask_bool, exclusion > 0)
    ))
    return overlap_area / mask_area


def clip_mask_to_allowed_region(
    mask_u8: np.ndarray,
) -> np.ndarray:
    """将 Mask 与允许区域取交集。"""
    if not ENABLE_RIGHT_EXCLUSION:
        return mask_u8

    height, width = mask_u8.shape[:2]
    allowed = build_allowed_region_mask(height, width)
    return cv2.bitwise_and(mask_u8, allowed)


def get_mask_bbox(
    mask: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def mask_iou(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_bool = first > 0
    second_bool = second > 0

    intersection = np.count_nonzero(
        np.logical_and(first_bool, second_bool)
    )

    union = np.count_nonzero(
        np.logical_or(first_bool, second_bool)
    )

    return (
        intersection / union
        if union > 0
        else 0.0
    )


def mask_containment(
    smaller: np.ndarray,
    larger: np.ndarray,
) -> float:
    smaller_bool = smaller > 0
    smaller_area = np.count_nonzero(smaller_bool)

    if smaller_area == 0:
        return 0.0

    intersection = np.count_nonzero(
        np.logical_and(
            smaller_bool,
            larger > 0,
        )
    )

    return intersection / smaller_area


def border_touch_ratio(mask: np.ndarray) -> float:
    binary = mask > 0
    height, width = binary.shape

    touched = (
        np.count_nonzero(binary[0, :])
        + np.count_nonzero(binary[height - 1, :])
        + np.count_nonzero(binary[:, 0])
        + np.count_nonzero(binary[:, width - 1])
    )

    border_size = 2 * height + 2 * width

    return touched / max(border_size, 1)


def keep_largest_component(
    mask_u8: np.ndarray,
) -> Tuple[np.ndarray, float]:
    binary = np.where(
        mask_u8 > 0,
        255,
        0,
    ).astype(np.uint8)

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    if count <= 1:
        return binary, 0.0

    areas = stats[
        1:,
        cv2.CC_STAT_AREA,
    ]

    largest_label = int(np.argmax(areas)) + 1
    largest_area = int(areas[largest_label - 1])
    total_area = int(np.count_nonzero(binary))

    result = np.where(
        labels == largest_label,
        255,
        0,
    ).astype(np.uint8)

    ratio = largest_area / max(total_area, 1)

    return result, ratio


def fill_internal_holes(
    mask_u8: np.ndarray,
) -> np.ndarray:
    binary = np.where(
        mask_u8 > 0,
        255,
        0,
    ).astype(np.uint8)

    height, width = binary.shape

    padded = cv2.copyMakeBorder(
        binary,
        1,
        1,
        1,
        1,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    flood = padded.copy()

    flood_mask = np.zeros(
        (
            flood.shape[0] + 2,
            flood.shape[1] + 2,
        ),
        dtype=np.uint8,
    )

    cv2.floodFill(
        flood,
        flood_mask,
        (0, 0),
        255,
    )

    flood = flood[
        1:height + 1,
        1:width + 1,
    ]

    holes = cv2.bitwise_not(flood)

    return cv2.bitwise_or(
        binary,
        holes,
    )


def mask_solidity(mask_u8: np.ndarray) -> float:
    points = cv2.findNonZero(
        np.where(
            mask_u8 > 0,
            255,
            0,
        ).astype(np.uint8)
    )

    if points is None:
        return 0.0

    hull = cv2.convexHull(points)
    hull_area = cv2.contourArea(hull)

    if hull_area <= 0:
        return 0.0

    area = np.count_nonzero(mask_u8 > 0)

    return float(area / hull_area)


# =========================================================
# 10. SAM
# =========================================================

def build_mask_generator(device: str):
    try:
        from segment_anything import (
            SamAutomaticMaskGenerator,
            sam_model_registry,
        )
    except ImportError as error:
        raise ImportError(
            "缺少 segment_anything。请安装：\n"
            "pip install git+https://github.com/"
            "facebookresearch/segment-anything.git"
        ) from error

    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SAM checkpoint 不存在：{SAM_CHECKPOINT}"
        )

    sam = sam_model_registry[
        SAM_MODEL_TYPE
    ](
        checkpoint=str(SAM_CHECKPOINT)
    )

    sam.to(device=device)
    sam.eval()

    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=SAM_POINTS_PER_SIDE,
        pred_iou_thresh=SAM_PRED_IOU_THRESH,
        stability_score_thresh=(
            SAM_STABILITY_SCORE_THRESH
        ),
        crop_n_layers=SAM_CROP_N_LAYERS,
        crop_n_points_downscale_factor=(
            SAM_CROP_N_POINTS_DOWNSCALE_FACTOR
        ),
        min_mask_region_area=(
            SAM_MIN_MASK_REGION_AREA
        ),
    )


# =========================================================
# 11. 餐具实例筛选
# =========================================================

def score_container_candidate(
    candidate: dict,
    image_height: int,
    image_width: int,
) -> Optional[ContainerInstance]:
    segmentation = candidate.get("segmentation")

    if segmentation is None:
        return None

    mask_u8 = (
        segmentation.astype(np.uint8)
        * 255
    )

    # 右侧裤腿等干扰区域中的候选不允许作为餐具。
    right_overlap = calculate_right_exclusion_overlap(mask_u8)

    if (
        ENABLE_RIGHT_EXCLUSION
        and right_overlap > MAX_CONTAINER_RIGHT_OVERLAP_RATIO
    ):
        return None

    # 双保险：裁掉禁区部分。
    mask_u8 = clip_mask_to_allowed_region(mask_u8)

    largest_mask, largest_ratio = (
        keep_largest_component(mask_u8)
    )

    if largest_ratio < MIN_LARGEST_COMPONENT_RATIO:
        return None

    largest_mask = fill_internal_holes(
        largest_mask
    )

    bbox = get_mask_bbox(largest_mask)

    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox

    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 2 or box_height <= 2:
        return None

    image_area = max(
        image_height * image_width,
        1,
    )

    area = int(
        np.count_nonzero(largest_mask)
    )

    area_ratio = area / image_area

    if not (
        MIN_CONTAINER_AREA_RATIO
        <= area_ratio
        <= MAX_CONTAINER_AREA_RATIO
    ):
        return None

    bbox_area_ratio = (
        box_width * box_height
        / image_area
    )

    if bbox_area_ratio > MAX_CONTAINER_BBOX_AREA_RATIO:
        return None

    aspect_ratio = (
        box_width
        / max(box_height, 1)
    )

    if not (
        MIN_CONTAINER_ASPECT_RATIO
        <= aspect_ratio
        <= MAX_CONTAINER_ASPECT_RATIO
    ):
        return None

    edge_ratio = border_touch_ratio(
        largest_mask
    )

    if edge_ratio > MAX_BORDER_TOUCH_RATIO:
        return None

    solidity = mask_solidity(
        largest_mask
    )

    if solidity < MIN_CONTAINER_SOLIDITY:
        return None

    compactness = (
        area
        / max(
            box_width * box_height,
            1,
        )
    )

    if compactness < MIN_CONTAINER_COMPACTNESS:
        return None

    predicted_iou = float(
        candidate.get("predicted_iou", 0.0)
    )

    stability = float(
        candidate.get("stability_score", 0.0)
    )

    preferred_area = 0.14

    size_score = math.exp(
        -abs(
            math.log(
                max(area_ratio, 1e-6)
                / preferred_area
            )
        )
    )

    score = (
        1.30 * predicted_iou
        + 1.10 * stability
        + 0.70 * largest_ratio
        + 0.45 * solidity
        + 0.30 * compactness
        + 0.35 * size_score
        - 1.60 * edge_ratio
    )

    return ContainerInstance(
        mask=largest_mask,
        bbox=bbox,
        score=score,
        area_ratio=area_ratio,
        predicted_iou=predicted_iou,
        stability_score=stability,
        solidity=solidity,
        compactness=compactness,
    )


def deduplicate_containers(
    candidates: List[ContainerInstance],
) -> List[ContainerInstance]:
    candidates = sorted(
        candidates,
        key=lambda item: (
            item.score,
            item.area_ratio,
        ),
        reverse=True,
    )

    kept: List[ContainerInstance] = []

    for candidate in candidates:
        duplicate = False

        for existing in kept:
            iou = mask_iou(
                candidate.mask,
                existing.mask,
            )

            candidate_in_existing = mask_containment(
                candidate.mask,
                existing.mask,
            )

            existing_in_candidate = mask_containment(
                existing.mask,
                candidate.mask,
            )

            if (
                iou >= CONTAINER_NMS_IOU_THRESHOLD
                or candidate_in_existing
                >= CONTAINER_CONTAINMENT_THRESHOLD
                or existing_in_candidate
                >= CONTAINER_CONTAINMENT_THRESHOLD
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

        if len(kept) >= MAX_CONTAINERS_PER_IMAGE:
            break

    if EXPECTED_CONTAINERS_PER_IMAGE > 0:
        kept = kept[
            :EXPECTED_CONTAINERS_PER_IMAGE
        ]

    kept.sort(
        key=lambda item: (
            item.bbox[1],
            item.bbox[0],
        )
    )

    return kept


def detect_containers(
    image_bgr: np.ndarray,
    mask_generator,
) -> List[ContainerInstance]:
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    raw_masks = mask_generator.generate(
        image_rgb
    )

    height, width = image_bgr.shape[:2]

    candidates = []

    for raw_mask in raw_masks:
        scored = score_container_candidate(
            raw_mask,
            height,
            width,
        )

        if scored is not None:
            candidates.append(scored)

    return deduplicate_containers(
        candidates
    )


# =========================================================
# 12. 从餐具 Mask 得到 inner_mask
# =========================================================

def build_inner_mask(
    outer_mask_u8: np.ndarray,
) -> Optional[np.ndarray]:
    outer = np.where(
        outer_mask_u8 > 0,
        255,
        0,
    ).astype(np.uint8)

    if OUTER_CLOSE_ITERATIONS > 0:
        kernel_size = max(
            1,
            int(OUTER_CLOSE_KERNEL),
        )

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.ones(
            (
                kernel_size,
                kernel_size,
            ),
            dtype=np.uint8,
        )

        outer = cv2.morphologyEx(
            outer,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=(
                OUTER_CLOSE_ITERATIONS
            ),
        )

    outer = fill_internal_holes(outer)

    outer_area = int(
        np.count_nonzero(outer)
    )

    if outer_area == 0:
        return None

    distance = cv2.distanceTransform(
        outer,
        cv2.DIST_L2,
        5,
    )

    max_distance = float(distance.max())

    if max_distance <= 0:
        return None

    threshold = max(
        float(INNER_MIN_SHRINK_PIXELS),
        max_distance * INNER_DISTANCE_RATIO,
    )

    inner = np.where(
        distance >= threshold,
        255,
        0,
    ).astype(np.uint8)

    inner, _ = keep_largest_component(inner)

    inner_area = int(
        np.count_nonzero(inner)
    )

    ratio = inner_area / max(outer_area, 1)

    if ratio < MIN_INNER_TO_OUTER_RATIO:
        threshold = max(
            1.0,
            max_distance * 0.10,
        )

        inner = np.where(
            distance >= threshold,
            255,
            0,
        ).astype(np.uint8)

        inner, _ = keep_largest_component(inner)

        inner_area = int(
            np.count_nonzero(inner)
        )

        ratio = inner_area / max(outer_area, 1)

    if not (
        MIN_INNER_TO_OUTER_RATIO
        <= ratio
        <= MAX_INNER_TO_OUTER_RATIO
    ):
        return None

    # 最终 inner_mask 不能进入右侧禁区。
    inner = clip_mask_to_allowed_region(inner)
    inner, _ = keep_largest_component(inner)

    if np.count_nonzero(inner) == 0:
        return None

    return inner


# =========================================================
# 13. Food RGBA 放置
# =========================================================

def crop_rgba_by_alpha(
    rgba: np.ndarray,
) -> Optional[np.ndarray]:
    if (
        rgba is None
        or rgba.ndim != 3
        or rgba.shape[2] != 4
    ):
        return None

    alpha = rgba[:, :, 3]

    bbox = get_mask_bbox(
        np.where(
            alpha >= SOURCE_ALPHA_THRESHOLD,
            255,
            0,
        ).astype(np.uint8)
    )

    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox

    pad = 1

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)

    x2 = min(
        rgba.shape[1],
        x2 + pad,
    )

    y2 = min(
        rgba.shape[0],
        y2 + pad,
    )

    return rgba[
        y1:y2,
        x1:x2,
    ].copy()


def normalize_axis_angle(
    angle_degrees: float,
) -> float:
    """将无方向长轴角度归一化到 [-90, 90)。"""
    angle = float(angle_degrees)

    while angle >= 90.0:
        angle -= 180.0

    while angle < -90.0:
        angle += 180.0

    return angle


def axis_angle_error(
    first_angle: float,
    second_angle: float,
) -> float:
    """计算两条无方向长轴之间的最小夹角。"""
    return abs(
        normalize_axis_angle(
            first_angle - second_angle
        )
    )


def calculate_axis_rotation(
    source_angle: float,
    target_angle: float,
) -> float:
    """
    将图像坐标系中的长轴角差转换为 OpenCV 实际旋转角。

    图像坐标 y 轴向下，而 cv2.getRotationMatrix2D 的正角方向
    与这里测得的图像角度符号相反，因此需要取负号。
    """
    image_delta = normalize_axis_angle(
        target_angle - source_angle
    )

    return -image_delta


def analyze_mask_shape(
    mask_u8: np.ndarray,
) -> Optional[dict]:
    """
    使用最小外接旋转矩形的最长边定义长轴。

    相比 PCA，这种方式对长方形和圆角长方形更稳定，
    不容易把短边误认为长边。
    """
    binary = np.where(
        mask_u8 > 0,
        255,
        0,
    ).astype(np.uint8)

    points = cv2.findNonZero(binary)

    if (
        points is None
        or len(points) < MIN_SHAPE_MASK_PIXELS
    ):
        return None

    rect = cv2.minAreaRect(points)
    rect_width = float(rect[1][0])
    rect_height = float(rect[1][1])

    if rect_width <= 1e-6 or rect_height <= 1e-6:
        return None

    box = cv2.boxPoints(rect).astype(np.float64)
    edges = []

    for index in range(4):
        start_point = box[index]
        end_point = box[(index + 1) % 4]
        vector = end_point - start_point
        length = float(np.linalg.norm(vector))

        if length <= 1e-6:
            continue

        angle = math.degrees(
            math.atan2(
                vector[1],
                vector[0],
            )
        )

        edges.append(
            (
                length,
                normalize_axis_angle(angle),
            )
        )

    if not edges:
        return None

    edges.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    long_side, long_axis_angle = edges[0]
    short_side = min(
        item[0]
        for item in edges
    )

    aspect_ratio = (
        long_side
        / max(short_side, 1e-6)
    )

    mask_area = float(
        np.count_nonzero(binary)
    )

    rect_area = rect_width * rect_height
    rectangularity = (
        mask_area
        / max(rect_area, 1e-6)
    )

    return {
        "aspect_ratio": float(aspect_ratio),
        "rectangularity": float(rectangularity),
        "long_axis_angle": float(long_axis_angle),
        "long_side": float(long_side),
        "short_side": float(short_side),
        "mask_area": float(mask_area),
    }


def rotate_rgba_bound(
    rgba: np.ndarray,
    angle_degrees: float,
) -> np.ndarray:
    """旋转 RGBA 并扩展画布，避免食物被裁掉。"""
    angle = float(angle_degrees)

    if abs(angle) < 0.25:
        return rgba.copy()

    height, width = rgba.shape[:2]
    center_x = width / 2.0
    center_y = height / 2.0

    matrix = cv2.getRotationMatrix2D(
        (center_x, center_y),
        angle,
        1.0,
    )

    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])

    new_width = max(
        1,
        int(math.ceil(height * sine + width * cosine)),
    )
    new_height = max(
        1,
        int(math.ceil(height * cosine + width * sine)),
    )

    matrix[0, 2] += new_width / 2.0 - center_x
    matrix[1, 2] += new_height / 2.0 - center_y

    rotated_bgr = cv2.warpAffine(
        rgba[:, :, :3],
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    rotated_alpha = cv2.warpAffine(
        rgba[:, :, 3],
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return np.dstack([
        rotated_bgr,
        rotated_alpha,
    ]).astype(np.uint8)


def align_rgba_long_axis_to_target(
    source_rgba: np.ndarray,
    source_shape: dict,
    target_shape: dict,
) -> Optional[dict]:
    """
    同时尝试多个旋转方向，旋转后重新测量长轴，
    选择真正与餐盘长轴最接近的结果。
    """
    primary_rotation = calculate_axis_rotation(
        source_shape["long_axis_angle"],
        target_shape["long_axis_angle"],
    )

    if VERIFY_LONG_AXIS_AFTER_ROTATION:
        candidate_rotations = [
            primary_rotation,
            -primary_rotation,
            primary_rotation + 90.0,
            primary_rotation - 90.0,
        ]
    else:
        candidate_rotations = [
            primary_rotation,
        ]

    unique_rotations = []

    for angle in candidate_rotations:
        normalized = normalize_axis_angle(angle)

        if not any(
            axis_angle_error(
                normalized,
                existing,
            ) < 0.25
            for existing in unique_rotations
        ):
            unique_rotations.append(normalized)

    best = None

    for rotation_degrees in unique_rotations:
        rotated = rotate_rgba_bound(
            source_rgba,
            rotation_degrees,
        )

        rotated = crop_rgba_by_alpha(rotated)

        if rotated is None:
            continue

        rotated_mask = np.where(
            rotated[:, :, 3]
            >= SOURCE_ALPHA_THRESHOLD,
            255,
            0,
        ).astype(np.uint8)

        rotated_shape = analyze_mask_shape(
            rotated_mask
        )

        if rotated_shape is None:
            continue

        error = axis_angle_error(
            rotated_shape["long_axis_angle"],
            target_shape["long_axis_angle"],
        )

        candidate = {
            "prepared_rgba": rotated,
            "prepared_shape": rotated_shape,
            "rotation_degrees": float(
                rotation_degrees
            ),
            "alignment_error": float(error),
        }

        if (
            best is None
            or candidate["alignment_error"]
            < best["alignment_error"]
        ):
            best = candidate

    if best is None:
        return None

    if (
        best["alignment_error"]
        > MAX_LONG_AXIS_ALIGNMENT_ERROR
    ):
        return None

    return best


def prepare_food_for_container_shape(
    source_rgba: np.ndarray,
    inner_mask_u8: np.ndarray,
) -> Optional[dict]:
    """
    形状兼容判断，并在需要时旋转食物。

    规则：
    1. 长方形/长条形食物只能进入长方形餐盘；
    2. 长边自动与餐盘长边对齐；
    3. 可选限制两者长短边比例差异。
    """
    source_mask = np.where(
        source_rgba[:, :, 3] >= SOURCE_ALPHA_THRESHOLD,
        255,
        0,
    ).astype(np.uint8)

    food_shape = analyze_mask_shape(source_mask)
    container_shape = analyze_mask_shape(inner_mask_u8)

    if food_shape is None or container_shape is None:
        return None

    food_is_rectangular = (
        food_shape['aspect_ratio'] >= FOOD_RECT_ASPECT_THRESHOLD
        and food_shape['rectangularity']
        >= FOOD_RECTANGULARITY_THRESHOLD
    )

    container_is_rectangular = (
        container_shape['aspect_ratio']
        >= CONTAINER_RECT_ASPECT_THRESHOLD
        and container_shape['rectangularity']
        >= CONTAINER_RECTANGULARITY_THRESHOLD
    )

    if (
        ENABLE_SHAPE_MATCHING
        and RECTANGULAR_FOOD_ONLY_TO_RECTANGULAR_CONTAINER
        and food_is_rectangular
        and not container_is_rectangular
    ):
        return None

    aspect_factor = 1.0

    if (
        ENABLE_SHAPE_MATCHING
        and ENABLE_RECT_ASPECT_RATIO_MATCH
        and food_is_rectangular
        and container_is_rectangular
    ):
        aspect_factor = max(
            food_shape['aspect_ratio']
            / max(container_shape['aspect_ratio'], 1e-6),
            container_shape['aspect_ratio']
            / max(food_shape['aspect_ratio'], 1e-6),
        )

        if aspect_factor > MAX_RECT_ASPECT_RATIO_FACTOR:
            return None

    rotation_degrees = 0.0
    alignment_error = 0.0
    prepared_rgba = source_rgba.copy()
    prepared_food_shape = food_shape

    if (
        ENABLE_SHAPE_MATCHING
        and ALIGN_RECTANGULAR_LONG_AXIS
        and food_is_rectangular
        and container_is_rectangular
    ):
        alignment = align_rgba_long_axis_to_target(
            source_rgba,
            food_shape,
            container_shape,
        )

        if alignment is None:
            return None

        prepared_rgba = alignment[
            "prepared_rgba"
        ]

        prepared_food_shape = alignment[
            "prepared_shape"
        ]

        rotation_degrees = alignment[
            "rotation_degrees"
        ]

        alignment_error = alignment[
            "alignment_error"
        ]

    return {
        "prepared_rgba": prepared_rgba,
        "food_shape": food_shape,
        "prepared_food_shape": prepared_food_shape,
        "container_shape": container_shape,
        "food_is_rectangular": food_is_rectangular,
        "container_is_rectangular": container_is_rectangular,
        "rotation_degrees": float(rotation_degrees),
        "alignment_error": float(alignment_error),
        "aspect_factor": float(aspect_factor),
    }


def build_offsets(
    inner_width: int,
    inner_height: int,
) -> List[Tuple[int, int]]:
    offsets = {(0, 0)}

    for ratio in FOOD_OFFSET_RATIOS:
        dx = int(round(
            inner_width * ratio
        ))

        dy = int(round(
            inner_height * ratio
        ))

        for offset_x in (-dx, 0, dx):
            for offset_y in (-dy, 0, dy):
                offsets.add(
                    (
                        offset_x,
                        offset_y,
                    )
                )

    return sorted(offsets)


def create_placed_food(
    food_rgba: np.ndarray,
    paste_x: int,
    paste_y: int,
    image_width: int,
    image_height: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    canvas = np.zeros(
        (
            image_height,
            image_width,
            4,
        ),
        dtype=np.uint8,
    )

    placed_binary = np.zeros(
        (
            image_height,
            image_width,
        ),
        dtype=np.uint8,
    )

    food_height, food_width = food_rgba.shape[:2]

    x1 = max(0, paste_x)
    y1 = max(0, paste_y)

    x2 = min(
        image_width,
        paste_x + food_width,
    )

    y2 = min(
        image_height,
        paste_y + food_height,
    )

    if x2 <= x1 or y2 <= y1:
        return canvas, placed_binary, 0

    source_x1 = x1 - paste_x
    source_y1 = y1 - paste_y
    source_x2 = source_x1 + (x2 - x1)
    source_y2 = source_y1 + (y2 - y1)

    patch = food_rgba[
        source_y1:source_y2,
        source_x1:source_x2,
    ].copy()

    binary_patch = (
        patch[:, :, 3]
        >= SOURCE_ALPHA_THRESHOLD
    )

    if USE_BINARY_SOURCE_ALPHA:
        patch[:, :, 3] = (
            binary_patch.astype(np.uint8)
            * 255
        )

    canvas[
        y1:y2,
        x1:x2,
    ] = patch

    placed_binary[
        y1:y2,
        x1:x2,
    ] = binary_patch.astype(np.uint8)

    visible_area = int(
        np.count_nonzero(placed_binary)
    )

    return (
        canvas,
        placed_binary,
        visible_area,
    )


def find_best_placement(
    source_rgba: np.ndarray,
    inner_mask_u8: np.ndarray,
) -> Optional[dict]:
    source = crop_rgba_by_alpha(
        source_rgba
    )

    if source is None:
        return None

    shape_result = prepare_food_for_container_shape(
        source,
        inner_mask_u8,
    )

    if shape_result is None:
        return None

    source = shape_result['prepared_rgba']

    source_height, source_width = source.shape[:2]

    if source_width <= 1 or source_height <= 1:
        return None

    inner_bbox = get_mask_bbox(inner_mask_u8)

    if inner_bbox is None:
        return None

    x1, y1, x2, y2 = inner_bbox

    inner_width = x2 - x1
    inner_height = y2 - y1

    inner_area = int(
        np.count_nonzero(inner_mask_u8)
    )

    if inner_area == 0:
        return None

    moments = cv2.moments(
        inner_mask_u8,
        binaryImage=True,
    )

    if moments["m00"] > 0:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
    else:
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

    if (
        shape_result["food_is_rectangular"]
        and shape_result["container_is_rectangular"]
    ):
        # 长方形严格按照长边对长边、短边对短边计算缩放。
        prepared_shape = shape_result[
            "prepared_food_shape"
        ]

        container_shape = shape_result[
            "container_shape"
        ]

        base_scale = min(
            container_shape["long_side"]
            / max(
                prepared_shape["long_side"],
                1e-6,
            ),
            container_shape["short_side"]
            / max(
                prepared_shape["short_side"],
                1e-6,
            ),
        )
    else:
        base_scale = min(
            inner_width / max(source_width, 1),
            inner_height / max(source_height, 1),
        )

    base_scale *= FOOD_BASE_FILL_RATIO

    offsets = build_offsets(
        inner_width,
        inner_height,
    )

    image_height, image_width = (
        inner_mask_u8.shape
    )

    best = None

    for multiplier in FOOD_SCALE_MULTIPLIERS:
        scale = base_scale * multiplier

        if (
            scale <= 0
            or scale > MAX_FOOD_UPSCALE
        ):
            continue

        new_width = max(
            1,
            int(round(source_width * scale)),
        )

        new_height = max(
            1,
            int(round(source_height * scale)),
        )

        interpolation = (
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_CUBIC
        )

        resized = cv2.resize(
            source,
            (new_width, new_height),
            interpolation=interpolation,
        )

        base_x = int(round(
            center_x - new_width / 2.0
        ))

        base_y = int(round(
            center_y - new_height / 2.0
        ))

        for offset_x, offset_y in offsets:
            paste_x = base_x + offset_x
            paste_y = base_y + offset_y

            (
                canvas,
                placed_binary,
                visible_area,
            ) = create_placed_food(
                resized,
                paste_x,
                paste_y,
                image_width,
                image_height,
            )

            if visible_area == 0:
                continue

            inside_area = int(
                np.count_nonzero(
                    np.logical_and(
                        placed_binary > 0,
                        inner_mask_u8 > 0,
                    )
                )
            )

            inside_ratio = (
                inside_area
                / max(visible_area, 1)
            )

            inner_coverage = (
                inside_area
                / max(inner_area, 1)
            )

            if inside_ratio < MIN_FOOD_INSIDE_RATIO:
                continue

            score = (
                1.8 * inside_ratio
                - 1.1 * abs(
                    inner_coverage
                    - TARGET_INNER_COVERAGE
                )
                + 0.10 * min(scale, 1.5)
            )

            result = {
                "canvas": canvas,
                "inside_ratio": inside_ratio,
                "inner_coverage": inner_coverage,
                "scale": scale,
                "paste_x": paste_x,
                "paste_y": paste_y,
                "score": score,
                "food_is_rectangular": shape_result["food_is_rectangular"],
                "container_is_rectangular": shape_result["container_is_rectangular"],
                "food_aspect_ratio": shape_result["food_shape"]["aspect_ratio"],
                "container_aspect_ratio": shape_result["container_shape"]["aspect_ratio"],
                "food_rectangularity": shape_result["food_shape"]["rectangularity"],
                "container_rectangularity": shape_result["container_shape"]["rectangularity"],
                "rotation_degrees": shape_result["rotation_degrees"],
                "alignment_error": shape_result["alignment_error"],
                "aspect_factor": shape_result["aspect_factor"],
            }

            if (
                best is None
                or score > best["score"]
            ):
                best = result

    return best


def blend_food_inside_inner_mask(
    base_bgr: np.ndarray,
    food_canvas_rgba: np.ndarray,
    inner_mask_u8: np.ndarray,
) -> np.ndarray:
    result = base_bgr.copy()

    # 最终融合前再次裁掉右侧禁区。
    inner_mask_u8 = clip_mask_to_allowed_region(inner_mask_u8)

    source_alpha = (
        food_canvas_rgba[:, :, 3]
    )

    binary = np.logical_and(
        source_alpha >= SOURCE_ALPHA_THRESHOLD,
        inner_mask_u8 > 0,
    ).astype(np.uint8)

    if np.count_nonzero(binary) == 0:
        return result

    if INWARD_FEATHER_PIXELS > 0:
        distance = cv2.distanceTransform(
            binary,
            cv2.DIST_L2,
            5,
        )

        alpha = np.clip(
            distance
            / max(
                INWARD_FEATHER_PIXELS,
                1e-6,
            ),
            0.0,
            1.0,
        )

        alpha *= binary.astype(np.float32)
    else:
        alpha = binary.astype(np.float32)

    if not USE_BINARY_SOURCE_ALPHA:
        alpha *= (
            source_alpha.astype(np.float32)
            / 255.0
        )

    alpha_3 = alpha[:, :, None]

    food_bgr = (
        food_canvas_rgba[:, :, :3]
        .astype(np.float32)
    )

    background = result.astype(np.float32)

    blended = (
        food_bgr * alpha_3
        + background * (1.0 - alpha_3)
    )

    return np.clip(
        blended,
        0,
        255,
    ).astype(np.uint8)


# =========================================================
# 14. 可视化
# =========================================================

def save_masks_and_visualization(
    source_image_path: Path,
    image_bgr: np.ndarray,
    containers: List[ContainerInstance],
    inner_masks: List[np.ndarray],
):
    stem = safe_relative_stem(
        source_image_path
    )

    visualization = image_bgr.copy()

    if ENABLE_RIGHT_EXCLUSION and DRAW_RIGHT_EXCLUSION_IN_VIS:
        image_height, image_width = visualization.shape[:2]
        start_x = get_right_exclusion_start_x(image_width)

        overlay = visualization.copy()
        overlay[:, start_x:] = (0, 0, 255)

        visualization = cv2.addWeighted(
            visualization,
            0.72,
            overlay,
            0.28,
            0.0,
        )

        cv2.line(
            visualization,
            (start_x, 0),
            (start_x, image_height - 1),
            (0, 0, 255),
            3,
        )

        cv2.putText(
            visualization,
            'RIGHT EXCLUSION',
            (min(start_x + 8, max(0, image_width - 190)), 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    for index, (
        container,
        inner_mask,
    ) in enumerate(
        zip(containers, inner_masks)
    ):
        outer_path = (
            OUTER_MASK_ROOT
            / f"{stem}__container"
            f"{index:02d}_outer.png"
        )

        inner_path = (
            INNER_MASK_ROOT
            / f"{stem}__container"
            f"{index:02d}_inner.png"
        )

        cv2.imwrite(
            str(outer_path),
            container.mask,
        )

        cv2.imwrite(
            str(inner_path),
            inner_mask,
        )

        outer_contours, _ = cv2.findContours(
            container.mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        inner_contours, _ = cv2.findContours(
            inner_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            visualization,
            outer_contours,
            -1,
            (0, 255, 0),
            2,
        )

        cv2.drawContours(
            visualization,
            inner_contours,
            -1,
            (0, 255, 255),
            2,
        )

        x1, y1, _, _ = container.bbox

        cv2.putText(
            visualization,
            f"container {index} score={container.score:.2f}",
            (
                x1,
                max(18, y1 - 6),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    vis_path = (
        VIS_ROOT
        / f"{stem}__containers.jpg"
    )

    cv2.imwrite(
        str(vis_path),
        visualization,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            92,
        ],
    )


# =========================================================
# 15. 四卡启动
# =========================================================

def launch_workers(args):
    gpu_ids = [
        item.strip()
        for item in args.gpu_ids.split(",")
        if item.strip()
    ]

    if len(gpu_ids) != 4:
        raise ValueError(
            "--launch_4gpu 需要 4 个 GPU ID，"
            "例如 0,1,2,3"
        )

    ensure_output_dirs()

    processes = []

    for rank, gpu_id in enumerate(gpu_ids):
        log_path = (
            LOG_ROOT
            / f"rank{rank}_gpu{gpu_id}.log"
        )

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--rank",
            str(rank),
            "--world_size",
            str(len(gpu_ids)),
            "--device",
            f"cuda:{gpu_id}",
        ]

        if args.max_images > 0:
            command.extend(
                [
                    "--max_images",
                    str(args.max_images),
                ]
            )

        log_file = log_path.open(
            "w",
            encoding="utf-8",
        )

        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        processes.append(
            (
                process,
                log_file,
                log_path,
            )
        )

        print(
            f"Started rank={rank}, "
            f"gpu={gpu_id}, "
            f"log={log_path}"
        )

    failures = []

    for process, log_file, log_path in processes:
        return_code = process.wait()
        log_file.close()

        if return_code != 0:
            failures.append(
                (
                    return_code,
                    log_path,
                )
            )

    if failures:
        message = "\n".join(
            f"return_code={code}, log={path}"
            for code, path in failures
        )

        raise RuntimeError(
            "部分 worker 失败：\n"
            + message
        )

    print("All workers finished.")


# =========================================================
# 16. 主流程
# =========================================================

def main():
    args = parse_args()

    if args.launch_4gpu:
        launch_workers(args)
        return

    validate_rank(
        args.rank,
        args.world_size,
    )

    ensure_output_dirs()

    if not EMPTY_IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"空碗目录不存在：{EMPTY_IMAGE_ROOT}"
        )

    if not FOOD_RGBA_ROOT.exists():
        raise FileNotFoundError(
            f"food_rgba 目录不存在：{FOOD_RGBA_ROOT}"
        )

    device = resolve_device(
        args.device,
        args.rank,
    )

    image_paths = collect_images(
        EMPTY_IMAGE_ROOT
    )

    if args.max_images > 0:
        image_paths = image_paths[
            :args.max_images
        ]

    total_before_shard = len(image_paths)

    image_paths = image_paths[
        args.rank::args.world_size
    ]

    food_paths = collect_food_rgba(
        FOOD_RGBA_ROOT
    )

    print("====================================")
    print("Paste food RGBA to empty bowls")
    print("====================================")
    print("Device:", device)
    print("Rank:", args.rank)
    print("World size:", args.world_size)
    print("Images before shard:", total_before_shard)
    print("Images for this rank:", len(image_paths))
    print("Food RGBA:", len(food_paths))
    print(
        "Expected containers per image:",
        EXPECTED_CONTAINERS_PER_IMAGE,
    )
    print("Right exclusion:", ENABLE_RIGHT_EXCLUSION)
    print("Right exclusion start ratio:", RIGHT_EXCLUSION_START_RATIO)
    print("Right exclusion start x:", RIGHT_EXCLUSION_START_X)
    print("Shape matching:", ENABLE_SHAPE_MATCHING)
    print(
        "Rectangular food only to rectangular container:",
        RECTANGULAR_FOOD_ONLY_TO_RECTANGULAR_CONTAINER,
    )
    print(
        "Align rectangular long axis:",
        ALIGN_RECTANGULAR_LONG_AXIS,
    )
    print(
        "Verify long axis after rotation:",
        VERIFY_LONG_AXIS_AFTER_ROTATION,
    )
    print(
        "Max long-axis alignment error:",
        MAX_LONG_AXIS_ALIGNMENT_ERROR,
    )
    print("====================================")

    if not image_paths:
        print("没有空碗图片。")
        return

    if not food_paths:
        print("没有找到 food_rgba。")
        return

    mask_generator = build_mask_generator(
        device
    )

    rng = random.Random(
        RANDOM_SEED + args.rank
    )

    meta_rows = []

    for image_path in tqdm(
        image_paths,
        desc=f"rank{args.rank}",
    ):
        image_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            continue

        try:
            containers = detect_containers(
                image_bgr,
                mask_generator,
            )
        except RuntimeError as error:
            print(
                f"[SAM失败] {image_path}: {error}"
            )
            continue

        if not containers:
            print(
                f"[未检测到碗] {image_path}"
            )
            continue

        valid_containers = []
        inner_masks = []

        for container in containers:
            inner_mask = build_inner_mask(
                container.mask
            )

            if inner_mask is None:
                continue

            valid_containers.append(container)
            inner_masks.append(inner_mask)

        if not valid_containers:
            print(
                f"[无有效inner_mask] {image_path}"
            )
            continue

        save_masks_and_visualization(
            image_path,
            image_bgr,
            valid_containers,
            inner_masks,
        )

        stem = safe_relative_stem(
            image_path
        )

        for variant_index in range(
            NUM_VARIANTS_PER_IMAGE
        ):
            output_path = (
                OUTPUT_IMAGE_ROOT
                / (
                    f"{stem}"
                    f"__food_variant"
                    f"{variant_index:02d}.jpg"
                )
            )

            if (
                SKIP_EXISTING_OUTPUT
                and output_path.exists()
            ):
                continue

            synthetic = image_bgr.copy()

            used_food_paths: Set[str] = set()
            records = []
            failed = False

            for container_index, (
                container,
                inner_mask,
            ) in enumerate(
                zip(
                    valid_containers,
                    inner_masks,
                )
            ):
                trial_count = min(
                    MAX_SOURCE_TRIALS,
                    len(food_paths),
                )

                trial_paths = rng.sample(
                    food_paths,
                    trial_count,
                )

                selected = None

                for food_path in trial_paths:
                    food_path_text = str(
                        food_path
                    )

                    if (
                        UNIQUE_FOOD_WITHIN_IMAGE
                        and food_path_text
                        in used_food_paths
                    ):
                        continue

                    source_rgba = cv2.imread(
                        str(food_path),
                        cv2.IMREAD_UNCHANGED,
                    )

                    if (
                        source_rgba is None
                        or source_rgba.ndim != 3
                        or source_rgba.shape[2] != 4
                    ):
                        continue

                    placement = find_best_placement(
                        source_rgba,
                        inner_mask,
                    )

                    if placement is None:
                        continue

                    selected = {
                        "food_path": food_path,
                        **placement,
                    }
                    break

                if selected is None:
                    if REQUIRE_ALL_CONTAINERS_FILLED:
                        failed = True
                        break

                    continue

                synthetic = blend_food_inside_inner_mask(
                    synthetic,
                    selected["canvas"],
                    inner_mask,
                )

                used_food_paths.add(
                    str(selected["food_path"])
                )

                x1, y1, x2, y2 = (
                    container.bbox
                )

                records.append(
                    [
                        str(image_path),
                        str(output_path),
                        variant_index,
                        container_index,
                        f"{x1},{y1},{x2},{y2}",
                        str(selected["food_path"]),
                        selected["inside_ratio"],
                        selected["inner_coverage"],
                        selected["scale"],
                        selected["food_is_rectangular"],
                        selected["container_is_rectangular"],
                        selected["food_aspect_ratio"],
                        selected["container_aspect_ratio"],
                        selected["food_rectangularity"],
                        selected["container_rectangularity"],
                        selected["rotation_degrees"],
                        selected["alignment_error"],
                        selected["aspect_factor"],
                    ]
                )

            if failed or not records:
                continue

            success = cv2.imwrite(
                str(output_path),
                synthetic,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY,
                ],
            )

            if success:
                meta_rows.extend(records)

    meta_path = (
        META_ROOT
        / f"paste_meta_rank"
        f"{args.rank}.csv"
    )

    with meta_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "source_empty_image",
                "output_image",
                "variant_index",
                "container_index",
                "container_bbox_xyxy",
                "food_rgba",
                "food_inside_ratio",
                "inner_coverage",
                "food_scale",
                "food_is_rectangular",
                "container_is_rectangular",
                "food_aspect_ratio",
                "container_aspect_ratio",
                "food_rectangularity",
                "container_rectangularity",
                "rotation_degrees",
                "long_axis_alignment_error",
                "aspect_ratio_factor",
            ]
        )

        writer.writerows(meta_rows)

    print("\nFinished!")
    print("Output images:", OUTPUT_IMAGE_ROOT)
    print("Visualizations:", VIS_ROOT)
    print("Outer masks:", OUTER_MASK_ROOT)
    print("Inner masks:", INNER_MASK_ROOT)
    print("Meta:", meta_path)


if __name__ == "__main__":
    main()
