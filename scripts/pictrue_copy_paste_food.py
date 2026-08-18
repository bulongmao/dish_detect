import argparse
import csv
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# =========================================================
# 1. 路径配置
# =========================================================

# 原始图片及YOLO标签
DATASET_ROOT = Path(
    "/data/ljy/dish_detect/lichu_dish_cls"
)

IMAGE_ROOT = DATASET_ROOT / "images"
LABEL_ROOT = DATASET_ROOT / "labels"

# Step1生成的plate crop
# 按你当前实际路径修改
CROP_ROOT = Path(
    "/data/ljy/dish_detect/lichu_dish_cls/plate_crops"
)

# Step2生成的SAM结果
SAM_ROOT = Path(
    "/data/ljy/dish_detect/sam"
)

MASK_ROOT = SAM_ROOT / "sam_food_masks"
RGBA_ROOT = SAM_ROOT / "sam_food_rgba"

# 输出目录
OUTPUT_ROOT = Path(
    "/data/ljy/dish_detect/result"
)

OUT_IMAGE_ROOT = OUTPUT_ROOT / "images"
OUT_LABEL_ROOT = OUTPUT_ROOT / "labels"


# =========================================================
# 2. 数据增强参数
# =========================================================

# 每张原始大图生成几个食物替换版本
NUM_SYN_PER_IMAGE = 3

# True：
#   可以使用相同盘子类别或不同盘子类别中的食物
#
# False：
#   只使用不同盘子类别中的食物
ALLOW_ANY_SOURCE = True

# 是否排除来自同一张原始大图的源食物
# 建议设为True，避免同图内部互换
EXCLUDE_SAME_ORIGINAL = True

# True：
#   一张图中的所有YOLO盘子都必须具备有效Mask，
#   否则整张原图不参与生成
REQUIRE_ALL_TARGETS_PREPARED = False

# True：
#   一张合成图中的所有目标盘子都必须替换成功，
#   否则该合成版本不保存
REQUIRE_ALL_PLATES_REPLACED = False

# 每张合成图片至少需要成功替换的餐盘数量
MIN_REPLACED_PLATES = 2

# 已有结果则跳过
SKIP_EXISTING = True

RANDOM_SEED = 42


# =========================================================
# 3. Mask质量参数
# =========================================================

# 源食物Mask面积占源crop面积的范围
# 太小：可能只提取了一个小碎片
# 太大：可能把盘子也提取进去了
MIN_SOURCE_MASK_RATIO = 0.05
MAX_SOURCE_MASK_RATIO = 0.60

# 目标Mask面积范围
# 目标Mask只用于计算原食物覆盖率
MIN_TARGET_MASK_RATIO = 0.05
MAX_TARGET_MASK_RATIO = 0.60

# 最关键参数：
# 新食物必须覆盖目标原食物Mask的最低比例
#
# 0.90表示覆盖至少90%
# 如果数据量允许，可提高到0.95
MIN_TARGET_COVERAGE = 0.90

# 缩放、裁剪后，源食物至少需要保留多少比例
# 防止大部分源食物被安全椭圆截掉
MIN_SOURCE_VISIBLE_RATIO = 0.75

# Alpha阈值
ALPHA_THRESHOLD = 128


# =========================================================
# 4. 缩放与融合参数
# =========================================================

# 新食物外接框最大可以占目标ROI宽高的比例
#
# 食物偏小可提高到0.92
# 食物贴到盘沿可降低到0.82
FILL_RATIO = 0.82

# 中心安全椭圆相对ROI的宽高比例
#
# 新食物只能出现在这个椭圆内部
SAFE_RATIO = 0.92

# 最大允许放大倍数
MAX_UPSCALE = 2.0

# 粘贴位置只允许在ROI中心附近小范围调整
# 用于提高对目标Mask的覆盖率
MAX_CENTER_OFFSET_RATIO = 0.05

# 向内羽化宽度
#
# 2：边缘较清晰
# 3～4：边缘更柔和
#
# 向内羽化不会产生Mask外侧的半透明光圈
FEATHER_WIDTH = 2.0

# 每个目标最多尝试多少个随机源食物
MAX_SOURCE_TRIALS = 500

# Step1若使用过bbox padding，这里必须保持相同
BBOX_PAD_RATIO = 0.0


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)


# =========================================================
# 5. 参数解析
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "在原始大图中同时替换多个盘子的食物，"
            "使用覆盖率验收，不生成Inpaint填充像素。"
        )
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="当前进程编号，从0开始",
    )

    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="并行进程总数",
    )

    return parser.parse_args()


# =========================================================
# 6. 通用工具
# =========================================================

def natural_sort_key(text: str):
    parts = re.split(r"(\d+)", text.lower())

    return [
        int(part) if part.isdigit() else part
        for part in parts
    ]


def validate_paths():
    required_dirs = [
        IMAGE_ROOT,
        LABEL_ROOT,
        CROP_ROOT,
        MASK_ROOT,
        RGBA_ROOT,
    ]

    for directory in required_dirs:
        if not directory.exists():
            raise FileNotFoundError(
                f"目录不存在：{directory}"
            )


def ensure_output_dirs():
    OUT_IMAGE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_LABEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


def build_original_image_index() -> Dict[str, Path]:
    """
    建立：
        图片stem -> 原始图片路径
    """
    image_index: Dict[str, Path] = {}

    for path in IMAGE_ROOT.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if path.stem in image_index:
            print(
                f"[警告] 原始图片stem重复：{path.stem}"
            )

        image_index[path.stem] = path

    return image_index


# =========================================================
# 7. crop文件名解析
# =========================================================

def parse_crop_filename(
    crop_path: Path,
) -> Optional[dict]:
    """
    Step1文件名格式：

        原图stem_plate类别_标签行序号.jpg

    例如：

        000123_plate4_2.jpg
    """
    match = re.match(
        r"^(.*)_plate(\d+)_(\d+)$",
        crop_path.stem,
    )

    if match is None:
        return None

    original_stem = match.group(1)
    plate_class = int(match.group(2))
    label_index = int(match.group(3))

    return {
        "original_stem": original_stem,
        "plate_class": plate_class,
        "plate_id": f"plate_{plate_class}",
        "label_index": label_index,
    }


# =========================================================
# 8. 构建有效样本列表
# =========================================================

def list_valid_items() -> List[dict]:
    """
    只有同时存在以下文件的样本才进入处理：

    1. plate crop
    2. SAM food mask
    3. SAM food RGBA
    4. 原始大图
    5. 原YOLO标签
    """
    image_index = build_original_image_index()

    items: List[dict] = []

    crop_paths = [
        path
        for path in CROP_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    crop_paths.sort(
        key=lambda path: natural_sort_key(
            str(path)
        )
    )

    for crop_path in crop_paths:
        parsed = parse_crop_filename(
            crop_path
        )

        if parsed is None:
            print(
                "[跳过] 无法解析crop文件名：",
                crop_path.name,
            )
            continue

        original_stem = parsed[
            "original_stem"
        ]

        original_image_path = image_index.get(
            original_stem
        )

        if original_image_path is None:
            continue

        label_path = (
            LABEL_ROOT
            / f"{original_stem}.txt"
        )

        if not label_path.exists():
            continue

        rel_path = crop_path.relative_to(
            CROP_ROOT
        )

        rel_dir = rel_path.parent
        base = crop_path.stem

        mask_path = (
            MASK_ROOT
            / rel_dir
            / f"{base}_food_mask.png"
        )

        rgba_path = (
            RGBA_ROOT
            / rel_dir
            / f"{base}_food_rgba.png"
        )

        if not mask_path.exists():
            continue

        if not rgba_path.exists():
            continue

        items.append(
            {
                "crop_path": crop_path,
                "mask_path": mask_path,
                "rgba_path": rgba_path,
                "original_image_path": original_image_path,
                "label_path": label_path,
                "original_stem": original_stem,
                "plate_class": parsed[
                    "plate_class"
                ],
                "plate_id": parsed[
                    "plate_id"
                ],
                "label_index": parsed[
                    "label_index"
                ],
                "base": base,
            }
        )

    return items


def group_items_by_original_image(
    items: List[dict],
) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}

    for item in items:
        original_stem = item[
            "original_stem"
        ]

        groups.setdefault(
            original_stem,
            [],
        ).append(item)

    for original_stem in groups:
        groups[original_stem].sort(
            key=lambda item: item[
                "label_index"
            ]
        )

    return groups


# =========================================================
# 9. YOLO标签处理
# =========================================================

def read_yolo_label_line(
    label_path: Path,
    label_index: int,
) -> Optional[
    Tuple[int, float, float, float, float]
]:
    try:
        with label_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            lines = file.readlines()
    except OSError:
        return None

    if (
        label_index < 0
        or label_index >= len(lines)
    ):
        return None

    line = lines[label_index].strip()

    if not line:
        return None

    values = line.split()

    if len(values) != 5:
        return None

    try:
        class_id = int(
            float(values[0])
        )

        cx, cy, bw, bh = map(
            float,
            values[1:],
        )
    except ValueError:
        return None

    return class_id, cx, cy, bw, bh


def count_yolo_objects(
    label_path: Path,
) -> int:
    """
    统计原图中的有效YOLO对象数量。
    """
    try:
        with label_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            lines = file.readlines()
    except OSError:
        return 0

    count = 0

    for line in lines:
        values = line.strip().split()

        if len(values) == 5:
            count += 1

    return count


def yolo_to_xyxy(
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    image_width: int,
    image_height: int,
    pad_ratio: float = BBOX_PAD_RATIO,
) -> Optional[Tuple[int, int, int, int]]:
    """
    尽量与Step1中的YOLO坐标转换方式保持一致。
    """
    box_width = bw * image_width
    box_height = bh * image_height

    center_x = cx * image_width
    center_y = cy * image_height

    pad_x = box_width * pad_ratio
    pad_y = box_height * pad_ratio

    x1 = int(
        center_x
        - box_width / 2
        - pad_x
    )

    y1 = int(
        center_y
        - box_height / 2
        - pad_y
    )

    x2 = int(
        center_x
        + box_width / 2
        + pad_x
    )

    y2 = int(
        center_y
        + box_height / 2
        + pad_y
    )

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(
        image_width - 1,
        x2,
    )

    y2 = min(
        image_height - 1,
        y2,
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


# =========================================================
# 10. Mask与RGBA工具
# =========================================================

def mask_area_ratio(
    mask_u8: np.ndarray,
) -> float:
    return float(
        np.count_nonzero(mask_u8)
        / max(mask_u8.size, 1)
    )


def get_mask_bbox(
    mask_u8: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask_u8 > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    )


def crop_rgba_by_alpha(
    rgba: np.ndarray,
) -> Optional[np.ndarray]:
    """
    根据Alpha通道裁掉透明外围。

    pad使用1，避免保留过多源盘子RGB像素。
    """
    if (
        rgba.ndim != 3
        or rgba.shape[2] != 4
    ):
        return None

    alpha = rgba[:, :, 3]

    bbox = get_mask_bbox(alpha)

    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox

    pad = 1

    height, width = alpha.shape

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)

    x2 = min(
        width - 1,
        x2 + pad,
    )

    y2 = min(
        height - 1,
        y2 + pad,
    )

    return rgba[
        y1:y2 + 1,
        x1:x2 + 1,
    ]


def build_safe_ellipse_mask(
    height: int,
    width: int,
    safe_ratio: float = SAFE_RATIO,
) -> np.ndarray:
    """
    构造餐具ROI中心安全椭圆。
    """
    safe_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    center = (
        width // 2,
        height // 2,
    )

    axes = (
        max(
            1,
            int(width * safe_ratio / 2),
        ),
        max(
            1,
            int(height * safe_ratio / 2),
        ),
    )

    cv2.ellipse(
        safe_mask,
        center,
        axes,
        0,
        0,
        360,
        255,
        -1,
    )

    return safe_mask


# =========================================================
# 11. 放置后的Alpha与覆盖率计算
# =========================================================

def place_alpha_on_target(
    food_rgba: np.ndarray,
    paste_xy: Tuple[int, int],
    target_height: int,
    target_width: int,
    safe_mask: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    将源食物Alpha虚拟放置到目标ROI中。

    返回：
    1. 放置后的二值Alpha画布
    2. 源食物可见比例
    """
    paste_x, paste_y = paste_xy

    food_height, food_width = (
        food_rgba.shape[:2]
    )

    placed_alpha = np.zeros(
        (target_height, target_width),
        dtype=np.uint8,
    )

    x1 = max(0, paste_x)
    y1 = max(0, paste_y)

    x2 = min(
        target_width,
        paste_x + food_width,
    )

    y2 = min(
        target_height,
        paste_y + food_height,
    )

    if x2 <= x1 or y2 <= y1:
        return placed_alpha, 0.0

    food_x1 = x1 - paste_x
    food_y1 = y1 - paste_y

    food_x2 = food_x1 + (x2 - x1)
    food_y2 = food_y1 + (y2 - y1)

    source_alpha_full = (
        food_rgba[:, :, 3]
        >= ALPHA_THRESHOLD
    )

    source_total_area = int(
        np.count_nonzero(
            source_alpha_full
        )
    )

    if source_total_area == 0:
        return placed_alpha, 0.0

    source_alpha_patch = (
        food_rgba[
            food_y1:food_y2,
            food_x1:food_x2,
            3,
        ]
        >= ALPHA_THRESHOLD
    )

    safe_patch = (
        safe_mask[y1:y2, x1:x2]
        > 0
    )

    valid_patch = np.logical_and(
        source_alpha_patch,
        safe_patch,
    )

    placed_alpha[
        y1:y2,
        x1:x2,
    ] = valid_patch.astype(
        np.uint8
    )

    visible_area = int(
        np.count_nonzero(
            placed_alpha
        )
    )

    visible_ratio = (
        visible_area
        / max(source_total_area, 1)
    )

    return placed_alpha, visible_ratio


def calculate_target_coverage(
    placed_alpha: np.ndarray,
    target_mask_u8: np.ndarray,
) -> float:
    """
    覆盖率定义：

        新食物与目标food mask交集面积
        --------------------------------
                 目标food mask面积
    """
    target_bool = (
        target_mask_u8 > 0
    )

    target_area = int(
        np.count_nonzero(
            target_bool
        )
    )

    if target_area == 0:
        return 0.0

    source_bool = (
        placed_alpha > 0
    )

    covered_area = int(
        np.count_nonzero(
            np.logical_and(
                target_bool,
                source_bool,
            )
        )
    )

    return (
        covered_area
        / target_area
    )


# =========================================================
# 12. 源食物缩放与中心放置
# =========================================================

def resize_and_place_food(
    source_rgba: np.ndarray,
    target_roi_bgr: np.ndarray,
    target_mask_u8: np.ndarray,
) -> Optional[dict]:
    """
    将源食物缩放到目标ROI允许的最大尺寸，
    并只在ROI中心附近小范围尝试位置。

    返回覆盖率最高的位置。
    """
    source_food = crop_rgba_by_alpha(
        source_rgba
    )

    if source_food is None:
        return None

    source_height, source_width = (
        source_food.shape[:2]
    )

    if source_width <= 1 or source_height <= 1:
        return None

    target_height, target_width = (
        target_roi_bgr.shape[:2]
    )

    safe_width = max(
        1,
        int(target_width * FILL_RATIO),
    )

    safe_height = max(
        1,
        int(target_height * FILL_RATIO),
    )

    box_scale = min(
        safe_width / source_width,
        safe_height / source_height,
    )

    # 在最大放大倍数范围内尽可能放大，
    # 以提高对原食物的覆盖率
    scale = min(
        box_scale,
        MAX_UPSCALE,
    )

    if scale <= 0:
        return None

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

    resized_food = cv2.resize(
        source_food,
        (new_width, new_height),
        interpolation=interpolation,
    )

    safe_mask = build_safe_ellipse_mask(
        target_height,
        target_width,
        SAFE_RATIO,
    )

    center_x = target_width // 2
    center_y = target_height // 2

    base_x = center_x - new_width // 2
    base_y = center_y - new_height // 2

    offset_x = max(
        1,
        int(
            target_width
            * MAX_CENTER_OFFSET_RATIO
        ),
    )

    offset_y = max(
        1,
        int(
            target_height
            * MAX_CENTER_OFFSET_RATIO
        ),
    )

    # 只允许在中心附近小幅移动，
    # 不会像旧版本一样跟着错误Mask移动到奇怪位置
    offsets = [
        (0, 0),
        (-offset_x, 0),
        (offset_x, 0),
        (0, -offset_y),
        (0, offset_y),
        (-offset_x, -offset_y),
        (offset_x, -offset_y),
        (-offset_x, offset_y),
        (offset_x, offset_y),
    ]

    best_result = None

    for dx, dy in offsets:
        paste_xy = (
            base_x + dx,
            base_y + dy,
        )

        placed_alpha, visible_ratio = (
            place_alpha_on_target(
                resized_food,
                paste_xy,
                target_height,
                target_width,
                safe_mask,
            )
        )

        coverage = calculate_target_coverage(
            placed_alpha,
            target_mask_u8,
        )

        candidate = {
            "resized_food": resized_food,
            "paste_xy": paste_xy,
            "coverage": coverage,
            "visible_ratio": visible_ratio,
            "scale": scale,
        }

        if best_result is None:
            best_result = candidate
            continue

        if coverage > best_result["coverage"]:
            best_result = candidate
        elif (
            coverage
            == best_result["coverage"]
            and visible_ratio
            > best_result["visible_ratio"]
        ):
            best_result = candidate

    if best_result is None:
        return None

    if (
        best_result["coverage"]
        < MIN_TARGET_COVERAGE
    ):
        return None

    if (
        best_result["visible_ratio"]
        < MIN_SOURCE_VISIBLE_RATIO
    ):
        return None

    return best_result


# =========================================================
# 13. 无外圈光晕的Alpha融合
# =========================================================

def alpha_blend_food(
    base_roi_bgr: np.ndarray,
    food_rgba: np.ndarray,
    paste_xy: Tuple[int, int],
) -> np.ndarray:
    """
    直接在原始ROI上覆盖食物。

    不使用Inpaint，不生成任何伪造盘底像素。

    使用向内羽化：
    - Mask外部Alpha始终为0；
    - 不会产生半透明外围光圈；
    - 只在食物内部边缘进行少量平滑。
    """
    result = base_roi_bgr.copy()

    target_height, target_width = (
        result.shape[:2]
    )

    paste_x, paste_y = paste_xy

    food_height, food_width = (
        food_rgba.shape[:2]
    )

    safe_mask = build_safe_ellipse_mask(
        target_height,
        target_width,
        SAFE_RATIO,
    )

    x1 = max(0, paste_x)
    y1 = max(0, paste_y)

    x2 = min(
        target_width,
        paste_x + food_width,
    )

    y2 = min(
        target_height,
        paste_y + food_height,
    )

    if x2 <= x1 or y2 <= y1:
        return result

    food_x1 = x1 - paste_x
    food_y1 = y1 - paste_y

    food_x2 = food_x1 + (x2 - x1)
    food_y2 = food_y1 + (y2 - y1)

    food_patch = food_rgba[
        food_y1:food_y2,
        food_x1:food_x2,
    ]

    food_bgr = (
        food_patch[:, :, :3]
        .astype(np.float32)
    )

    source_alpha = (
        food_patch[:, :, 3]
    )

    safe_patch = (
        safe_mask[y1:y2, x1:x2]
    )

    binary_mask = np.logical_and(
        source_alpha
        >= ALPHA_THRESHOLD,
        safe_patch > 0,
    ).astype(np.uint8)

    if np.count_nonzero(binary_mask) == 0:
        return result

    # 向内距离变换
    distance = cv2.distanceTransform(
        binary_mask,
        cv2.DIST_L2,
        5,
    )

    inward_alpha = np.clip(
        distance
        / max(FEATHER_WIDTH, 1e-6),
        0.0,
        1.0,
    )

    # 保留源Alpha自身的有效程度，
    # 同时严格限制Mask外部为0
    original_alpha = (
        source_alpha.astype(np.float32)
        / 255.0
    )

    alpha = (
        inward_alpha
        * original_alpha
        * binary_mask.astype(np.float32)
    )

    alpha = np.clip(
        alpha,
        0.0,
        1.0,
    )

    alpha_3 = alpha[:, :, None]

    background_patch = (
        result[y1:y2, x1:x2]
        .astype(np.float32)
    )

    blended = (
        food_bgr * alpha_3
        + background_patch
        * (1.0 - alpha_3)
    )

    result[y1:y2, x1:x2] = (
        blended.astype(np.uint8)
    )

    return result


# =========================================================
# 14. 源食物选择
# =========================================================

def source_is_allowed(
    source: dict,
    target: dict,
    used_source_paths: Set[str],
) -> bool:
    source_path = str(
        source["crop_path"]
    )

    if (
        source["crop_path"]
        == target["crop_path"]
    ):
        return False

    if source_path in used_source_paths:
        return False

    if (
        EXCLUDE_SAME_ORIGINAL
        and source["original_stem"]
        == target["original_stem"]
    ):
        return False

    if (
        not ALLOW_ANY_SOURCE
        and source["plate_id"]
        == target["plate_id"]
    ):
        return False

    return True


def select_valid_source(
    items: List[dict],
    target: dict,
    target_roi: np.ndarray,
    target_mask: np.ndarray,
    used_source_paths: Set[str],
) -> Optional[dict]:
    """
    随机尝试多个源食物。

    只有同时满足：
    1. 源Mask比例合格；
    2. 对目标Mask覆盖率合格；
    3. 源食物保留比例合格；

    才返回该源食物。
    """
    if not items:
        return None

    trial_count = min(
        MAX_SOURCE_TRIALS,
        len(items),
    )

    trial_indices = random.sample(
        range(len(items)),
        trial_count,
    )

    for index in trial_indices:
        source = items[index]

        if not source_is_allowed(
            source,
            target,
            used_source_paths,
        ):
            continue

        source_rgba = cv2.imread(
            str(source["rgba_path"]),
            cv2.IMREAD_UNCHANGED,
        )

        if source_rgba is None:
            continue

        if (
            source_rgba.ndim != 3
            or source_rgba.shape[2] != 4
        ):
            continue

        source_alpha = (
            source_rgba[:, :, 3]
        )

        source_ratio = mask_area_ratio(
            source_alpha
        )

        if not (
            MIN_SOURCE_MASK_RATIO
            <= source_ratio
            <= MAX_SOURCE_MASK_RATIO
        ):
            continue

        placement = resize_and_place_food(
            source_rgba,
            target_roi,
            target_mask,
        )

        if placement is None:
            continue

        return {
            "source": source,
            "source_ratio": source_ratio,
            **placement,
        }

    return None


# =========================================================
# 15. 目标盘子预处理
# =========================================================

def prepare_target_plates(
    target_group: List[dict],
    original_image: np.ndarray,
) -> List[dict]:
    """
    获取同一张原图中每个目标盘子的：

    - bbox
    - 原始ROI
    - 目标food mask
    - target mask面积比例

    注意：
    不生成empty ROI，不调用Inpaint。
    """
    image_height, image_width = (
        original_image.shape[:2]
    )

    prepared_targets = []

    for target in target_group:
        yolo_label = read_yolo_label_line(
            target["label_path"],
            target["label_index"],
        )

        if yolo_label is None:
            continue

        (
            label_class,
            cx,
            cy,
            bw,
            bh,
        ) = yolo_label

        if label_class != target[
            "plate_class"
        ]:
            continue

        bbox = yolo_to_xyxy(
            cx,
            cy,
            bw,
            bh,
            image_width,
            image_height,
        )

        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox

        target_roi = original_image[
            y1:y2,
            x1:x2,
        ].copy()

        if target_roi.size == 0:
            continue

        target_mask = cv2.imread(
            str(target["mask_path"]),
            cv2.IMREAD_GRAYSCALE,
        )

        if target_mask is None:
            continue

        roi_height, roi_width = (
            target_roi.shape[:2]
        )

        if target_mask.shape != (
            roi_height,
            roi_width,
        ):
            target_mask = cv2.resize(
                target_mask,
                (roi_width, roi_height),
                interpolation=cv2.INTER_NEAREST,
            )

        target_ratio = mask_area_ratio(
            target_mask
        )

        if not (
            MIN_TARGET_MASK_RATIO
            <= target_ratio
            <= MAX_TARGET_MASK_RATIO
        ):
            continue

        prepared_targets.append(
            {
                "target": target,
                "bbox": bbox,
                "target_roi": target_roi,
                "target_mask": target_mask,
                "target_ratio": target_ratio,
            }
        )

    return prepared_targets


# =========================================================
# 16. 保存
# =========================================================

def save_full_image(
    output_path: Path,
    image_bgr: np.ndarray,
) -> bool:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return bool(
        cv2.imwrite(
            str(output_path),
            image_bgr,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )
    )


# =========================================================
# 17. 主流程
# =========================================================

def main():
    args = parse_args()

    if args.rank < 0:
        raise ValueError(
            "rank不能小于0"
        )

    if args.world_size <= 0:
        raise ValueError(
            "world_size必须大于0"
        )

    if args.rank >= args.world_size:
        raise ValueError(
            "rank必须小于world_size"
        )

    validate_paths()
    ensure_output_dirs()

    random.seed(
        RANDOM_SEED + args.rank
    )

    items = list_valid_items()

    print("====================================")
    print("Coverage-controlled Copy-Paste")
    print("====================================")
    print("Valid plate items:", len(items))
    print("Rank:", args.rank)
    print("World size:", args.world_size)
    print("NUM_SYN_PER_IMAGE:", NUM_SYN_PER_IMAGE)
    print("MIN_TARGET_COVERAGE:", MIN_TARGET_COVERAGE)
    print("FILL_RATIO:", FILL_RATIO)
    print("SAFE_RATIO:", SAFE_RATIO)
    print("MAX_UPSCALE:", MAX_UPSCALE)
    print("REQUIRE_ALL_TARGETS_PREPARED:", REQUIRE_ALL_TARGETS_PREPARED)
    print("REQUIRE_ALL_PLATES_REPLACED:", REQUIRE_ALL_PLATES_REPLACED)
    print("====================================")

    if not items:
        print("没有找到有效样本。")
        return

    groups = group_items_by_original_image(
        items
    )

    image_groups = sorted(
        groups.items(),
        key=lambda pair: natural_sort_key(
            pair[0]
        ),
    )

    print(
        "Total original image groups:",
        len(image_groups),
    )

    # 按原始图片分片
    image_groups = image_groups[
        args.rank::args.world_size
    ]

    print(
        "Original images for this rank:",
        len(image_groups),
    )

    meta_rows = []

    for (
        original_stem,
        target_group,
    ) in tqdm(image_groups):

        if not target_group:
            continue

        first_target = target_group[0]

        original_image = cv2.imread(
            str(
                first_target[
                    "original_image_path"
                ]
            )
        )

        if original_image is None:
            continue

        expected_plate_count = (
            count_yolo_objects(
                first_target["label_path"]
            )
        )

        # target_group只包含已有SAM Mask和RGBA的盘子
        if (
            REQUIRE_ALL_TARGETS_PREPARED
            and len(target_group)
            != expected_plate_count
        ):
            continue

        prepared_targets = prepare_target_plates(
            target_group,
            original_image,
        )

        # 可处理餐盘不足2个，不需要继续尝试生成
        if len(prepared_targets) < MIN_REPLACED_PLATES:
            continue

        if (
            REQUIRE_ALL_TARGETS_PREPARED
            and len(prepared_targets)
            != expected_plate_count
        ):
            continue
        
        for syn_index in range(NUM_SYN_PER_IMAGE):
            output_stem = (
                f"{original_stem}"
                f"__coverage_replace"
                f"__syn{syn_index}"
            )

            output_image_path = (
                OUT_IMAGE_ROOT
                / f"{output_stem}.jpg"
            )

            output_label_path = (
                OUT_LABEL_ROOT
                / f"{output_stem}.txt"
            )

            if (
                SKIP_EXISTING
                and output_image_path.exists()
                and output_label_path.exists()
            ):
                continue

            # 每个版本都从原始真实图片开始
            synthetic_full = original_image.copy()

            used_source_paths: Set[str] = set()

            replacement_records = []
            replaced_count = 0

            for prepared in prepared_targets:
                target = prepared["target"]

                selected = select_valid_source(
                    items,
                    target,
                    prepared["target_roi"],
                    prepared["target_mask"],
                    used_source_paths,
                )

                # 当前餐盘失败不影响其他餐盘
                if selected is None:
                    continue

                synthetic_roi = alpha_blend_food(
                    prepared["target_roi"],
                    selected["resized_food"],
                    selected["paste_xy"],
                )

                x1, y1, x2, y2 = prepared["bbox"]

                synthetic_full[
                    y1:y2,
                    x1:x2,
                ] = synthetic_roi

                source = selected["source"]

                used_source_paths.add(
                    str(source["crop_path"])
                )

                replacement_records.append(
                    {
                        "target_plate_id": target[
                            "plate_id"
                        ],
                        "target_label_index": target[
                            "label_index"
                        ],
                        "target_crop": str(
                            target["crop_path"]
                        ),
                        "source_plate_id": source[
                            "plate_id"
                        ],
                        "source_crop": str(
                            source["crop_path"]
                        ),
                        "target_mask_ratio": prepared[
                            "target_ratio"
                        ],
                        "source_mask_ratio": selected[
                            "source_ratio"
                        ],
                        "target_coverage": selected[
                            "coverage"
                        ],
                        "source_visible_ratio": selected[
                            "visible_ratio"
                        ],
                        "source_scale": selected[
                            "scale"
                        ],
                    }
                )

                replaced_count += 1

            # 核心条件：
            # 至少成功替换两个餐盘才输出
            if replaced_count < MIN_REPLACED_PLATES:
                continue

            success = save_full_image(
                output_image_path,
                synthetic_full,
            )

            if not success:
                continue

            shutil.copy2(
                first_target["label_path"],
                output_label_path,
            )

            for record in replacement_records:
                meta_rows.append(
                    [
                        str(output_image_path),
                        str(output_label_path),
                        str(
                            first_target[
                                "original_image_path"
                            ]
                        ),
                        replaced_count,
                        expected_plate_count,
                        syn_index,
                        record["target_plate_id"],
                        record["target_label_index"],
                        record["target_crop"],
                        record["source_plate_id"],
                        record["source_crop"],
                        record["target_mask_ratio"],
                        record["source_mask_ratio"],
                        record["target_coverage"],
                        record["source_visible_ratio"],
                        record["source_scale"],
                    ]
                )
        

    meta_path = (
        OUTPUT_ROOT
        / f"result_meta_rank{args.rank}.csv"
    )

    with meta_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            [
                "output_image",
                "output_label",
                "target_original_image",
                "replaced_plate_count",
                "expected_plate_count",
                "synthetic_index",
                "target_plate_id",
                "target_label_index",
                "target_crop",
                "source_plate_id",
                "source_crop",
                "target_mask_ratio",
                "source_mask_ratio",
                "target_coverage",
                "source_visible_ratio",
                "source_scale",
            ]
        )

        writer.writerows(
            meta_rows
        )

    print("\nFinished!")
    print("Images:", OUT_IMAGE_ROOT)
    print("Labels:", OUT_LABEL_ROOT)
    print("Meta:", meta_path)
    print(
        "Meta rows generated:",
        len(meta_rows),
    )


if __name__ == "__main__":
    main()