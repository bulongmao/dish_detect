import argparse
import csv
import hashlib
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# =========================================================
# 1. 路径配置
# =========================================================

# 目标餐盘crop目录。目录中应包含plate_0～plate_13等子目录。
CROP_ROOT = Path(
    "/data/ljy/dish_detect/lichu_dish_cls/plate_crops"
)

# SAM生成的食物mask和RGBA目录。
SAM_ROOT = Path(
    "/data/ljy/dish_detect/sam"
)
MASK_ROOT = SAM_ROOT / "sam_food_masks"
RGBA_ROOT = SAM_ROOT / "sam_food_rgba"

# 输出目录：会自动创建plate_0～plate_13。
OUTPUT_ROOT = Path(
    "/data/ljy/dish_detect/result_per_plate_20"
)


# =========================================================
# 2. 生成模式
# =========================================================

# "per_class"：
#   每个盘子类别总共输出20张，最终通常为14×20=280张。
#   这是本脚本的默认模式。
#
# "per_target"：
#   每一张目标crop都输出20张。若每类有很多crop，输出量会很大。
GENERATION_MODE = "per_class"

# 每个盘子类别输出数量（per_class模式使用）。
NUM_VARIANTS_PER_CLASS = 20

# 每一张目标crop输出数量（per_target模式使用）。
NUM_VARIANTS_PER_TARGET = 20

# 盘子类别数量：输出plate_0～plate_13。
NUM_PLATE_CLASSES = 15

# True：只有凑齐要求数量才写出该组结果。
# False：找不到足够源食物时，也保存已经成功生成的部分。
REQUIRE_EXACT_COUNT = True

# 已经完整生成的组直接跳过。
SKIP_COMPLETE_GROUPS = True

# 随机种子。每个类别/目标都会基于该值产生独立且可复现的顺序。
RANDOM_SEED = 42


# =========================================================
# 3. 源食物选择规则
# =========================================================

# True：源食物可以来自任意盘子类别。
# False：源食物必须来自不同盘子类别。
ALLOW_ANY_SOURCE = True

# 排除与目标crop来自同一张原图的源食物。
EXCLUDE_SAME_ORIGINAL = True

# 每一个输出最多检查多少个随机源食物。
MAX_SOURCE_TRIALS = 1000

# per_class模式下，为生成一个结果最多更换多少个目标crop尝试。
MAX_TARGET_TRIALS_PER_VARIANT = 50

# 同一类别/同一目标的20张结果中，不重复使用同一个源图片。
REQUIRE_UNIQUE_SOURCE_PATH = True

# 使用dHash降低“不同文件但视觉几乎相同”的重复菜品。
USE_DHASH_DIVERSITY = True

# 64位dHash之间至少相差多少位才认为足够不同。
# 8～12通常比较合适；越大越严格，越难凑齐20张。
MIN_DHASH_DISTANCE = 10


# =========================================================
# 4. Mask质量参数
# =========================================================

MIN_SOURCE_MASK_RATIO = 0.05
MAX_SOURCE_MASK_RATIO = 0.60

MIN_TARGET_MASK_RATIO = 0.05
MAX_TARGET_MASK_RATIO = 0.60

# 新食物至少覆盖目标原食物mask的比例。
MIN_TARGET_COVERAGE = 0.90

# 新食物经过目标ROI边界和安全椭圆裁切后，至少保留的比例。
MIN_SOURCE_VISIBLE_RATIO = 0.75

ALPHA_THRESHOLD = 128


# =========================================================
# 5. 缩放与融合参数
# =========================================================

# 新食物最大外接框占目标crop宽高的比例。
FILL_RATIO = 0.83

# 目标crop中心安全椭圆比例。
SAFE_RATIO = 0.94

# 最大放大倍数。
MAX_UPSCALE = 2.5

# 为提高覆盖率，允许在中心附近尝试的小范围偏移。
MAX_CENTER_OFFSET_RATIO = 0.06

# 向内羽化宽度；不会向mask外扩散出半透明光圈。
FEATHER_WIDTH = 2.0


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)


# =========================================================
# 6. 参数解析
# =========================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "按14个盘子类别输出不同食物Copy-Paste结果，"
            "默认每类20张。"
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
# 7. 通用工具
# =========================================================


def natural_sort_key(text: str):
    parts = re.split(r"(\d+)", text.lower())
    return [
        int(part) if part.isdigit() else part
        for part in parts
    ]


def stable_seed(key: str) -> int:
    text = f"{RANDOM_SEED}|{key}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def validate_paths():
    for directory in [CROP_ROOT, MASK_ROOT, RGBA_ROOT]:
        if not directory.exists():
            raise FileNotFoundError(f"目录不存在：{directory}")


def ensure_output_dirs():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for class_id in range(NUM_PLATE_CLASSES):
        (OUTPUT_ROOT / f"plate_{class_id}").mkdir(
            parents=True,
            exist_ok=True,
        )

    classes_path = OUTPUT_ROOT / "classes.txt"
    if not classes_path.exists():
        classes_path.write_text(
            "\n".join(
                f"plate_{class_id}"
                for class_id in range(NUM_PLATE_CLASSES)
            )
            + "\n",
            encoding="utf-8",
        )


# =========================================================
# 8. crop文件解析和有效样本列表
# =========================================================


def parse_crop_filename(crop_path: Path) -> Optional[dict]:
    """
    文件名格式：
        原图stem_plate类别_标签行序号.jpg

    示例：
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


def list_valid_items() -> List[dict]:
    """
    只有同时存在以下文件的crop才进入素材池：
    1. plate crop；
    2. SAM food mask；
    3. SAM food RGBA。
    """
    crop_paths = [
        path
        for path in CROP_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    crop_paths.sort(
        key=lambda path: natural_sort_key(str(path))
    )

    items: List[dict] = []

    for crop_path in crop_paths:
        parsed = parse_crop_filename(crop_path)

        if parsed is None:
            print("[跳过] 无法解析crop文件名：", crop_path.name)
            continue

        class_id = parsed["plate_class"]

        if not (0 <= class_id < NUM_PLATE_CLASSES):
            continue

        rel_path = crop_path.relative_to(CROP_ROOT)
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

        if not mask_path.exists() or not rgba_path.exists():
            continue

        items.append(
            {
                "crop_path": crop_path,
                "mask_path": mask_path,
                "rgba_path": rgba_path,
                "original_stem": parsed["original_stem"],
                "plate_class": class_id,
                "plate_id": parsed["plate_id"],
                "label_index": parsed["label_index"],
                "base": base,
            }
        )

    return items


def group_items_by_plate_class(
    items: List[dict],
) -> Dict[int, List[dict]]:
    groups: Dict[int, List[dict]] = {
        class_id: []
        for class_id in range(NUM_PLATE_CLASSES)
    }

    for item in items:
        groups[item["plate_class"]].append(item)

    for class_id in groups:
        groups[class_id].sort(
            key=lambda item: natural_sort_key(item["base"])
        )

    return groups


# =========================================================
# 9. Mask和RGBA工具
# =========================================================


def mask_area_ratio(mask_u8: np.ndarray) -> float:
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
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        return None

    alpha = rgba[:, :, 3]
    bbox = get_mask_bbox(alpha)

    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    height, width = alpha.shape
    pad = 1

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width - 1, x2 + pad)
    y2 = min(height - 1, y2 + pad)

    return rgba[y1:y2 + 1, x1:x2 + 1]


def build_safe_ellipse_mask(
    height: int,
    width: int,
    safe_ratio: float = SAFE_RATIO,
) -> np.ndarray:
    safe_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    center = (width // 2, height // 2)
    axes = (
        max(1, int(width * safe_ratio / 2)),
        max(1, int(height * safe_ratio / 2)),
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
# 10. 覆盖率计算
# =========================================================


def place_alpha_on_target(
    food_rgba: np.ndarray,
    paste_xy: Tuple[int, int],
    target_height: int,
    target_width: int,
    safe_mask: np.ndarray,
) -> Tuple[np.ndarray, float]:
    paste_x, paste_y = paste_xy
    food_height, food_width = food_rgba.shape[:2]

    placed_alpha = np.zeros(
        (target_height, target_width),
        dtype=np.uint8,
    )

    x1 = max(0, paste_x)
    y1 = max(0, paste_y)
    x2 = min(target_width, paste_x + food_width)
    y2 = min(target_height, paste_y + food_height)

    if x2 <= x1 or y2 <= y1:
        return placed_alpha, 0.0

    food_x1 = x1 - paste_x
    food_y1 = y1 - paste_y
    food_x2 = food_x1 + (x2 - x1)
    food_y2 = food_y1 + (y2 - y1)

    source_alpha_full = (
        food_rgba[:, :, 3] >= ALPHA_THRESHOLD
    )
    source_total_area = int(
        np.count_nonzero(source_alpha_full)
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

    safe_patch = safe_mask[y1:y2, x1:x2] > 0
    valid_patch = np.logical_and(
        source_alpha_patch,
        safe_patch,
    )

    placed_alpha[y1:y2, x1:x2] = (
        valid_patch.astype(np.uint8)
    )

    visible_area = int(np.count_nonzero(placed_alpha))
    visible_ratio = visible_area / max(source_total_area, 1)

    return placed_alpha, visible_ratio


def calculate_target_coverage(
    placed_alpha: np.ndarray,
    target_mask_u8: np.ndarray,
) -> float:
    target_bool = target_mask_u8 > 0
    target_area = int(np.count_nonzero(target_bool))

    if target_area == 0:
        return 0.0

    source_bool = placed_alpha > 0
    covered_area = int(
        np.count_nonzero(
            np.logical_and(target_bool, source_bool)
        )
    )

    return covered_area / target_area


# =========================================================
# 11. 源食物缩放和放置
# =========================================================


def resize_and_place_food(
    source_rgba: np.ndarray,
    target_crop_bgr: np.ndarray,
    target_mask_u8: np.ndarray,
) -> Optional[dict]:
    source_food = crop_rgba_by_alpha(source_rgba)

    if source_food is None:
        return None

    source_height, source_width = source_food.shape[:2]

    if source_width <= 1 or source_height <= 1:
        return None

    target_height, target_width = target_crop_bgr.shape[:2]

    safe_width = max(1, int(target_width * FILL_RATIO))
    safe_height = max(1, int(target_height * FILL_RATIO))

    box_scale = min(
        safe_width / source_width,
        safe_height / source_height,
    )

    scale = min(box_scale, MAX_UPSCALE)

    if scale <= 0:
        return None

    new_width = max(1, int(round(source_width * scale)))
    new_height = max(1, int(round(source_height * scale)))

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
        int(target_width * MAX_CENTER_OFFSET_RATIO),
    )
    offset_y = max(
        1,
        int(target_height * MAX_CENTER_OFFSET_RATIO),
    )

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
        paste_xy = (base_x + dx, base_y + dy)

        placed_alpha, visible_ratio = place_alpha_on_target(
            resized_food,
            paste_xy,
            target_height,
            target_width,
            safe_mask,
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
            coverage == best_result["coverage"]
            and visible_ratio > best_result["visible_ratio"]
        ):
            best_result = candidate

    if best_result is None:
        return None

    if best_result["coverage"] < MIN_TARGET_COVERAGE:
        return None

    if best_result["visible_ratio"] < MIN_SOURCE_VISIBLE_RATIO:
        return None

    return best_result


# =========================================================
# 12. 向内羽化融合
# =========================================================


def alpha_blend_food(
    base_crop_bgr: np.ndarray,
    food_rgba: np.ndarray,
    paste_xy: Tuple[int, int],
) -> np.ndarray:
    result = base_crop_bgr.copy()

    target_height, target_width = result.shape[:2]
    paste_x, paste_y = paste_xy
    food_height, food_width = food_rgba.shape[:2]

    safe_mask = build_safe_ellipse_mask(
        target_height,
        target_width,
        SAFE_RATIO,
    )

    x1 = max(0, paste_x)
    y1 = max(0, paste_y)
    x2 = min(target_width, paste_x + food_width)
    y2 = min(target_height, paste_y + food_height)

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

    food_bgr = food_patch[:, :, :3].astype(np.float32)
    source_alpha = food_patch[:, :, 3]
    safe_patch = safe_mask[y1:y2, x1:x2]

    binary_mask = np.logical_and(
        source_alpha >= ALPHA_THRESHOLD,
        safe_patch > 0,
    ).astype(np.uint8)

    if np.count_nonzero(binary_mask) == 0:
        return result

    distance = cv2.distanceTransform(
        binary_mask,
        cv2.DIST_L2,
        5,
    )

    inward_alpha = np.clip(
        distance / max(FEATHER_WIDTH, 1e-6),
        0.0,
        1.0,
    )

    original_alpha = source_alpha.astype(np.float32) / 255.0

    alpha = (
        inward_alpha
        * original_alpha
        * binary_mask.astype(np.float32)
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha_3 = alpha[:, :, None]

    background_patch = result[y1:y2, x1:x2].astype(np.float32)

    blended = (
        food_bgr * alpha_3
        + background_patch * (1.0 - alpha_3)
    )

    result[y1:y2, x1:x2] = blended.astype(np.uint8)
    return result


# =========================================================
# 13. 视觉差异dHash
# =========================================================


def food_dhash(food_rgba: np.ndarray) -> int:
    """
    对RGBA食物生成64位dHash。
    透明区域填充为中灰色，避免透明背景干扰。
    """
    bgr = food_rgba[:, :, :3].copy()
    alpha = food_rgba[:, :, 3]
    bgr[alpha < ALPHA_THRESHOLD] = 127

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(
        gray,
        (9, 8),
        interpolation=cv2.INTER_AREA,
    )

    differences = small[:, 1:] > small[:, :-1]

    value = 0
    for bit in differences.flatten():
        value = (value << 1) | int(bit)

    return value


def hamming_distance(hash1: int, hash2: int) -> int:
    return (hash1 ^ hash2).bit_count()


def is_hash_diverse(
    candidate_hash: int,
    selected_hashes: List[int],
) -> bool:
    if not USE_DHASH_DIVERSITY:
        return True

    return all(
        hamming_distance(candidate_hash, existing_hash)
        >= MIN_DHASH_DISTANCE
        for existing_hash in selected_hashes
    )


# =========================================================
# 14. 目标准备和源食物选择
# =========================================================


def prepare_target_item(item: dict) -> Optional[dict]:
    target_crop = cv2.imread(str(item["crop_path"]))

    if target_crop is None or target_crop.size == 0:
        return None

    target_mask = cv2.imread(
        str(item["mask_path"]),
        cv2.IMREAD_GRAYSCALE,
    )

    if target_mask is None:
        return None

    target_height, target_width = target_crop.shape[:2]

    if target_mask.shape != (target_height, target_width):
        target_mask = cv2.resize(
            target_mask,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )

    target_ratio = mask_area_ratio(target_mask)

    if not (
        MIN_TARGET_MASK_RATIO
        <= target_ratio
        <= MAX_TARGET_MASK_RATIO
    ):
        return None

    return {
        "target": item,
        "target_crop": target_crop,
        "target_mask": target_mask,
        "target_ratio": target_ratio,
    }


def source_is_allowed(
    source: dict,
    target: dict,
    used_source_paths: Set[str],
) -> bool:
    source_path = str(source["crop_path"])

    if source["crop_path"] == target["crop_path"]:
        return False

    if (
        REQUIRE_UNIQUE_SOURCE_PATH
        and source_path in used_source_paths
    ):
        return False

    if (
        EXCLUDE_SAME_ORIGINAL
        and source["original_stem"] == target["original_stem"]
    ):
        return False

    if (
        not ALLOW_ANY_SOURCE
        and source["plate_class"] == target["plate_class"]
    ):
        return False

    return True


def select_valid_source(
    items: List[dict],
    prepared_target: dict,
    used_source_paths: Set[str],
    selected_hashes: List[int],
    rng: random.Random,
) -> Optional[dict]:
    target = prepared_target["target"]

    allowed_indices = [
        index
        for index, source in enumerate(items)
        if source_is_allowed(
            source,
            target,
            used_source_paths,
        )
    ]

    if not allowed_indices:
        return None

    rng.shuffle(allowed_indices)
    allowed_indices = allowed_indices[:MAX_SOURCE_TRIALS]

    for index in allowed_indices:
        source = items[index]

        source_rgba = cv2.imread(
            str(source["rgba_path"]),
            cv2.IMREAD_UNCHANGED,
        )

        if (
            source_rgba is None
            or source_rgba.ndim != 3
            or source_rgba.shape[2] != 4
        ):
            continue

        source_ratio = mask_area_ratio(
            source_rgba[:, :, 3]
        )

        if not (
            MIN_SOURCE_MASK_RATIO
            <= source_ratio
            <= MAX_SOURCE_MASK_RATIO
        ):
            continue

        placement = resize_and_place_food(
            source_rgba,
            prepared_target["target_crop"],
            prepared_target["target_mask"],
        )

        if placement is None:
            continue

        candidate_hash = food_dhash(
            placement["resized_food"]
        )

        if not is_hash_diverse(
            candidate_hash,
            selected_hashes,
        ):
            continue

        return {
            "source": source,
            "source_ratio": source_ratio,
            "food_hash": candidate_hash,
            **placement,
        }

    return None


# =========================================================
# 15. 生成一组20张结果
# =========================================================


def generate_variants_for_group(
    group_key: str,
    target_items: List[dict],
    source_items: List[dict],
    required_count: int,
) -> List[dict]:
    """
    group_key可以是plate_0，也可以是某个target base。

    返回的每个元素包含：
    - 合成后的crop；
    - 目标/源信息；
    - 覆盖率等元数据。
    """
    prepared_targets = []

    for item in target_items:
        prepared = prepare_target_item(item)
        if prepared is not None:
            prepared_targets.append(prepared)

    if not prepared_targets:
        return []

    rng = random.Random(stable_seed(group_key))
    rng.shuffle(prepared_targets)

    used_source_paths: Set[str] = set()
    selected_hashes: List[int] = []
    variants: List[dict] = []

    # 为避免某个目标crop不适配，允许轮换尝试同类的其他目标crop。
    for variant_index in range(required_count):
        selected_variant = None

        target_indices = list(range(len(prepared_targets)))
        rng.shuffle(target_indices)
        target_indices = target_indices[
            :MAX_TARGET_TRIALS_PER_VARIANT
        ]

        for target_index in target_indices:
            prepared_target = prepared_targets[target_index]

            selected = select_valid_source(
                source_items,
                prepared_target,
                used_source_paths,
                selected_hashes,
                rng,
            )

            if selected is None:
                continue

            synthetic_crop = alpha_blend_food(
                prepared_target["target_crop"],
                selected["resized_food"],
                selected["paste_xy"],
            )

            selected_variant = {
                "synthetic_crop": synthetic_crop,
                "target": prepared_target["target"],
                "target_mask_ratio": prepared_target["target_ratio"],
                "source": selected["source"],
                "source_mask_ratio": selected["source_ratio"],
                "target_coverage": selected["coverage"],
                "source_visible_ratio": selected["visible_ratio"],
                "source_scale": selected["scale"],
                "food_hash": selected["food_hash"],
            }
            break

        if selected_variant is None:
            break

        used_source_paths.add(
            str(selected_variant["source"]["crop_path"])
        )
        selected_hashes.append(
            selected_variant["food_hash"]
        )
        variants.append(selected_variant)

    return variants


# =========================================================
# 16. 输出文件名和保存
# =========================================================


def expected_output_paths(
    class_id: int,
    group_name: str,
    count: int,
) -> List[Path]:
    class_dir = OUTPUT_ROOT / f"plate_{class_id}"

    return [
        class_dir / f"{group_name}__food_{index:02d}.jpg"
        for index in range(count)
    ]


def save_variants(
    class_id: int,
    group_name: str,
    variants: List[dict],
) -> List[dict]:
    class_dir = OUTPUT_ROOT / f"plate_{class_id}"
    class_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for index, variant in enumerate(variants):
        output_path = (
            class_dir
            / f"{group_name}__food_{index:02d}.jpg"
        )

        success = cv2.imwrite(
            str(output_path),
            variant["synthetic_crop"],
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        if not success:
            print("[保存失败]", output_path)
            continue

        target = variant["target"]
        source = variant["source"]

        rows.append(
            {
                "output_path": str(output_path),
                "plate_class": class_id,
                "plate_id": f"plate_{class_id}",
                "group_name": group_name,
                "variant_index": index,
                "target_crop": str(target["crop_path"]),
                "target_original_stem": target["original_stem"],
                "target_label_index": target["label_index"],
                "source_crop": str(source["crop_path"]),
                "source_original_stem": source["original_stem"],
                "source_plate_class": source["plate_class"],
                "target_mask_ratio": variant["target_mask_ratio"],
                "source_mask_ratio": variant["source_mask_ratio"],
                "target_coverage": variant["target_coverage"],
                "source_visible_ratio": variant["source_visible_ratio"],
                "source_scale": variant["source_scale"],
                "food_dhash": str(variant["food_hash"]),
            }
        )

    return rows


# =========================================================
# 17. 主流程
# =========================================================


def main():
    args = parse_args()

    if args.world_size <= 0:
        raise ValueError("world_size必须大于0")

    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError("rank必须满足0 <= rank < world_size")

    if GENERATION_MODE not in {"per_class", "per_target"}:
        raise ValueError(
            "GENERATION_MODE只能是'per_class'或'per_target'"
        )

    validate_paths()
    ensure_output_dirs()

    items = list_valid_items()

    print("====================================")
    print("Per-plate food Copy-Paste")
    print("====================================")
    print("Mode:", GENERATION_MODE)
    print("Valid items:", len(items))
    print("Plate classes:", NUM_PLATE_CLASSES)
    print("Variants per class:", NUM_VARIANTS_PER_CLASS)
    print("Variants per target:", NUM_VARIANTS_PER_TARGET)
    print("Require exact count:", REQUIRE_EXACT_COUNT)
    print("dHash diversity:", USE_DHASH_DIVERSITY)
    print("Rank/world_size:", args.rank, "/", args.world_size)
    print("Output:", OUTPUT_ROOT)
    print("====================================")

    if not items:
        print("没有找到有效crop、mask和RGBA组合。")
        return

    groups_by_class = group_items_by_plate_class(items)
    meta_rows: List[dict] = []

    if GENERATION_MODE == "per_class":
        work_units = [
            (class_id, groups_by_class[class_id])
            for class_id in range(NUM_PLATE_CLASSES)
        ]

        work_units = work_units[
            args.rank::args.world_size
        ]

        for class_id, target_items in tqdm(
            work_units,
            desc=f"rank{args.rank}",
        ):
            group_name = f"plate_{class_id}"
            required_count = NUM_VARIANTS_PER_CLASS

            if not target_items:
                print(f"[跳过] plate_{class_id}没有有效目标crop")
                continue

            expected_paths = expected_output_paths(
                class_id,
                group_name,
                required_count,
            )

            if (
                SKIP_COMPLETE_GROUPS
                and all(path.exists() for path in expected_paths)
            ):
                continue

            variants = generate_variants_for_group(
                group_name,
                target_items,
                items,
                required_count,
            )

            if REQUIRE_EXACT_COUNT and len(variants) < required_count:
                print(
                    f"[未输出] plate_{class_id}只找到"
                    f"{len(variants)}/{required_count}个合格且不同的源食物"
                )
                continue

            rows = save_variants(
                class_id,
                group_name,
                variants,
            )
            meta_rows.extend(rows)

            print(
                f"[完成] plate_{class_id}: "
                f"输出{len(rows)}张"
            )

    else:
        target_items = sorted(
            items,
            key=lambda item: natural_sort_key(item["base"]),
        )

        target_items = target_items[
            args.rank::args.world_size
        ]

        for target_item in tqdm(
            target_items,
            desc=f"rank{args.rank}",
        ):
            class_id = target_item["plate_class"]
            group_name = target_item["base"]
            required_count = NUM_VARIANTS_PER_TARGET

            expected_paths = expected_output_paths(
                class_id,
                group_name,
                required_count,
            )

            if (
                SKIP_COMPLETE_GROUPS
                and all(path.exists() for path in expected_paths)
            ):
                continue

            variants = generate_variants_for_group(
                group_name,
                [target_item],
                items,
                required_count,
            )

            if REQUIRE_EXACT_COUNT and len(variants) < required_count:
                print(
                    f"[未输出] {group_name}只找到"
                    f"{len(variants)}/{required_count}个合格且不同的源食物"
                )
                continue

            rows = save_variants(
                class_id,
                group_name,
                variants,
            )
            meta_rows.extend(rows)

    meta_path = (
        OUTPUT_ROOT
        / f"result_meta_rank{args.rank}.csv"
    )

    fieldnames = [
        "output_path",
        "plate_class",
        "plate_id",
        "group_name",
        "variant_index",
        "target_crop",
        "target_original_stem",
        "target_label_index",
        "source_crop",
        "source_original_stem",
        "source_plate_class",
        "target_mask_ratio",
        "source_mask_ratio",
        "target_coverage",
        "source_visible_ratio",
        "source_scale",
        "food_dhash",
    ]

    with meta_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(meta_rows)

    print("\nFinished!")
    print("Output:", OUTPUT_ROOT)
    print("Meta:", meta_path)
    print("Generated rows:", len(meta_rows))


if __name__ == "__main__":
    main()
