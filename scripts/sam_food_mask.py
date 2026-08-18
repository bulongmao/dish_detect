import os
import csv
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# =========================================================
# 配置区域
# =========================================================

DATA_ROOT = "/data/ljy/dish_detect/extract"
OUTPUT_ROOT = "/data/ljy/dish_detect/sam_e"
SAM_ROOT = "/data/ljy/dish_detect"

CROP_ROOT = "/data/ljy/dish_detect/extract"
# CROP_ROOT = os.path.join(DATA_ROOT, "plate_crops")

CHECKPOINT = os.path.join(
    SAM_ROOT,
    "checkpoints",
    "sam_vit_h_4b8939.pth"
)

MODEL_TYPE = "vit_h"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 输出目录
MASK_ROOT = os.path.join(OUTPUT_ROOT, "sam_food_masks")
VIS_ROOT = os.path.join(OUTPUT_ROOT, "sam_food_vis")
FOOD_RGBA_ROOT = os.path.join(OUTPUT_ROOT, "sam_food_rgba")
META_CSV = os.path.join(OUTPUT_ROOT, "sam_food_mask_meta.csv")


# 是否只测试少量图片
# 第一次建议设置为 50，确认效果后再改成 None 跑全部
TEST_LIMIT = 50

# 是否启用多个候选 mask 合并
# False：只取最优 mask，更稳定
# True：合并多个中心区域 mask，可能覆盖更完整食物，但也更容易误选
USE_UNION_TOPK = True

# 跳过
SKIP_EXISTING = True
# 防止只选到一小块食物
MIN_ELLIPSE_COVERAGE = 0.08
# 如果最终best mask面积太小，尝试从候选中选一个个更大的
MIN_BEST_AREA_RATIO= 0.12
# 合并时最多合并几个候选 mask
UNION_TOPK = 5


# 食物 mask 面积约束
# 面积太小通常是碎片，面积太大通常是整个碗/盘
MIN_AREA_RATIO = 0.20
MAX_AREA_RATIO = 0.55

# 中心区域约束
# 食物一般位于碗/盘中心，边缘大面积贴边的 mask 更可能是餐具本身
MIN_CENTER_OVERLAP = 0.30
MAX_BORDER_TOUCH_RATIO = 0.24


# =========================================================
# 工具函数
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_images(root):
    image_paths = []

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(valid_exts):
                image_paths.append(
                    os.path.join(dirpath, name)
                )

    image_paths.sort()
    return image_paths


def make_center_ellipse(h, w):
    """
    构造中心椭圆区域。
    食物通常位于餐具中心，用它评估候选 mask 是否像食物。
    """
    ellipse = np.zeros((h, w), dtype=np.uint8)

    center = (w // 2, h // 2)

    # 椭圆轴长不要太大，否则会把碗边也纳入中心区域
    axes = (
        max(1, int(w * 0.36)),
        max(1, int(h * 0.36))
    )

    cv2.ellipse(
        ellipse,
        center,
        axes,
        0,
        0,
        360,
        1,
        -1
    )

    return ellipse.astype(bool)

def choose_best_candidate(candidates):
    """
    在原始版本基础上做温和改进：
    - 默认仍然选择 score 最高的 mask；
    - 如果 score 最高的 mask 面积太小，则尝试选择一个面积更合理、
      中心覆盖更完整、贴边不严重的候选。
    """

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = candidates[0]


    if (
        MIN_BEST_AREA_RATIO
        <= best["area_ratio"]
        <= MAX_AREA_RATIO
    ):
        return best

    # 如果 best 太小，尝试找一个更大的候选
    fallback_candidates = []

    for cand in candidates:
        if cand["area_ratio"] < MIN_BEST_AREA_RATIO:
            continue

        if cand["area_ratio"] > MAX_AREA_RATIO:
            continue

        if cand["border_ratio"] > MAX_BORDER_TOUCH_RATIO:
            continue

        if cand["ellipse_coverage"] < MIN_ELLIPSE_COVERAGE:
            continue

        fallback_candidates.append(cand)

    if len(fallback_candidates) == 0:
        return best

    # 在更大候选中，优先选中心覆盖率高、面积适中的
    fallback_candidates.sort(
        key=lambda x: (
            2.0 * x["ellipse_coverage"]
            + 0.8 * x["center_overlap"]
            + 0.5 * x["pred_iou"]
            - 0.6 * abs(x["area_ratio"] - 0.38)
            - 1.0 * x["border_ratio"]
        ),
        reverse=True
    )

    return fallback_candidates[0]

def border_touch_ratio(mask):
    """
    计算 mask 接触图像边界的比例。
    如果大量贴边，通常不是食物，而是整个餐具或背景。
    """
    h, w = mask.shape

    border_pixels = (
        np.count_nonzero(mask[0, :]) +
        np.count_nonzero(mask[h - 1, :]) +
        np.count_nonzero(mask[:, 0]) +
        np.count_nonzero(mask[:, w - 1])
    )

    total_border = 2 * h + 2 * w

    return border_pixels / max(total_border, 1)

def score_mask(mask_info, h, w, center_ellipse):
    """
    根据面积、中心性、中心椭圆覆盖率、贴边程度、SAM置信度综合评分。

    目标：
    1. 避免只选中中心一小块食物；
    2. 避免把碗边、外部容器、背景带进去；
    3. 保持原始版本“只选一个best mask”的稳定性。
    """
    mask = mask_info["segmentation"].astype(bool)

    img_area = h * w
    area = np.count_nonzero(mask)
    area_ratio = area / max(img_area, 1)

    if area_ratio < MIN_AREA_RATIO:
        return None

    if area_ratio > MAX_AREA_RATIO:
        return None

    border_ratio = border_touch_ratio(mask)

    if border_ratio > MAX_BORDER_TOUCH_RATIO:
        return None

    center_intersection = np.logical_and(
        mask,
        center_ellipse
    ).sum()

    center_overlap = center_intersection / max(area, 1)

    ellipse_area = center_ellipse.sum()
    ellipse_coverage = center_intersection / max(ellipse_area, 1)

    if center_overlap < MIN_CENTER_OVERLAP:
        return None

    # 关键：防止只选到中心一小块
    if ellipse_coverage < MIN_ELLIPSE_COVERAGE:
        return None

    pred_iou = float(mask_info.get("predicted_iou", 0.0))
    stability = float(mask_info.get("stability_score", 0.0))

    # 原来偏向 0.28，容易选小块；这里调到 0.38，更适合一碗食物
    size_penalty = abs(area_ratio - 0.34)

    # 小于10%的候选额外扣分
    small_area_penalty = max(
        0.0,
        0.10 - area_ratio
    )

    # 大于48%的候选额外重扣分
    large_area_penalty = max(
        0.0,
        area_ratio - 0.48
    )

    score = (
        1.0 * center_overlap
        + 0.8 * ellipse_coverage
        + 0.7 * pred_iou
        + 0.5 * stability
        - 1.2 * border_ratio
        - 0.6 * size_penalty
        - 2.0 * small_area_penalty
        - 5.0 * large_area_penalty
    )

    return {
        "mask": mask,
        "score": score,
        "area_ratio": area_ratio,
        "center_overlap": center_overlap,
        "ellipse_coverage": ellipse_coverage,
        "border_ratio": border_ratio,
        "pred_iou": pred_iou,
        "stability": stability
    }

def postprocess_mask(mask):
    """
    对 mask 做简单后处理：
    1. 转 uint8
    2. 闭运算填小孔
    3. 开运算去小噪声
    """
    mask_u8 = mask.astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    return mask_u8


def overlay_mask(image, mask_u8):
    """
    可视化：在原 crop 上覆盖红色 food mask。
    """
    vis = image.copy()

    red = np.zeros_like(image)
    red[:, :, 2] = 255

    alpha = 0.45
    mask_bool = mask_u8 > 0

    vis[mask_bool] = (
        image[mask_bool] * (1 - alpha)
        + red[mask_bool] * alpha
    ).astype(np.uint8)

    return vis


def save_food_rgba(image, mask_u8, save_path):
    """
    保存食物前景 RGBA 图：
    RGB 为原图像素，Alpha 为 food mask。
    """
    b, g, r = cv2.split(image)

    alpha = mask_u8

    rgba = cv2.merge([b, g, r, alpha])

    cv2.imwrite(save_path, rgba)


def build_output_paths(img_path, output_root, suffix):
    """
    保持 plate_x 子目录结构。
    输入：
        plate_crops/plate_4/xxx.jpg
    输出：
        sam_food_masks/plate_4/xxx_mask.png
    """
    rel_path = os.path.relpath(img_path, CROP_ROOT)
    rel_dir = os.path.dirname(rel_path)

    base_name = os.path.splitext(
        os.path.basename(img_path)
    )[0]

    out_dir = os.path.join(output_root, rel_dir)
    ensure_dir(out_dir)

    return os.path.join(out_dir, base_name + suffix)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Current worker rank."
    )

    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Total number of workers."
    )

    return parser.parse_args()

# =========================================================
# 主流程
# =========================================================

def main():

    args = parse_args()
    print("====================================")
    print("SAM food mask generation")
    print("====================================")
    print("CROP_ROOT:", CROP_ROOT)
    print("CHECKPOINT:", CHECKPOINT)
    print("MODEL_TYPE:", MODEL_TYPE)
    print("DEVICE:", DEVICE)
    print("TEST_LIMIT:", TEST_LIMIT)
    print("USE_UNION_TOPK:", USE_UNION_TOPK)
    print("====================================")

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"SAM checkpoint not found: {CHECKPOINT}"
        )

    ensure_dir(MASK_ROOT)
    ensure_dir(VIS_ROOT)
    ensure_dir(FOOD_RGBA_ROOT)

    image_paths = list_images(CROP_ROOT)

    # 如果测试，则先截取测试数量
    if TEST_LIMIT is not None:
        image_paths = image_paths[:TEST_LIMIT]

    total_before_shard = len(image_paths)

    # 多进程分片：rank 0 处理 0,4,8...；rank 1 处理 1,5,9...
    image_paths = image_paths[args.rank::args.world_size]

    print("Total crop images before shard:", total_before_shard)
    print("Rank:", args.rank)
    print("World size:", args.world_size)
    print("Images for this rank:", len(image_paths))

    # 加载 SAM
    sam = sam_model_registry[MODEL_TYPE](
        checkpoint=CHECKPOINT
    )

    sam.to(device=DEVICE)

    # 自动 mask 生成器
    # points_per_side 越大，mask 越细，但速度越慢
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.90,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100
    )

    meta_rows = []
    for img_path in tqdm(image_paths):

        # 先构建输出路径，支持断点续跑
        mask_path = build_output_paths(
            img_path,
            MASK_ROOT,
            "_food_mask.png"
        )

        vis_path = build_output_paths(
            img_path,
            VIS_ROOT,
            "_food_mask_vis.jpg"
        )

        rgba_path = build_output_paths(
            img_path,
            FOOD_RGBA_ROOT,
            "_food_rgba.png"
        )

        # 已经生成过 mask 和 rgba 的样本跳过
        if SKIP_EXISTING and os.path.exists(mask_path) and os.path.exists(rgba_path):
            meta_rows.append([
                img_path,
                "skipped_existing",
                0,
                0,
                0,
                0,
                0,
                0
            ])
            continue

        image_bgr = cv2.imread(img_path)

        if image_bgr is None:
            print("Cannot read:", img_path)
            continue

        h, w = image_bgr.shape[:2]

        if h < 20 or w < 20:
            print("Too small:", img_path)
            continue

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        center_ellipse = make_center_ellipse(h, w)

        try:
            masks = mask_generator.generate(image_rgb)
        except RuntimeError as e:
            print("SAM failed:", img_path)
            print(e)
            continue

        candidates = []

        for m in masks:
            s = score_mask(
                m,
                h,
                w,
                center_ellipse
            )

            if s is not None:
                candidates.append(s)

        if len(candidates) == 0:
            # 没有找到合适食物mask
            meta_rows.append([
                img_path,
                "failed",
                0,
                0,
                0,
                0,
                0,
                0
            ])
            continue

        best = choose_best_candidate(candidates)
        final_mask = best["mask"].copy()

        mask_u8 = postprocess_mask(final_mask)

        # 保存 mask
        cv2.imwrite(mask_path, mask_u8)

        # 保存可视化
        vis = overlay_mask(image_bgr, mask_u8)
        cv2.imwrite(vis_path, vis)

        # 保存食物前景RGBA
        save_food_rgba(
            image_bgr,
            mask_u8,
            rgba_path
        )

        meta_rows.append([
            img_path,
            "success",
            len(masks),
            len(candidates),
            best["score"],
            best["area_ratio"],
            best["center_overlap"],
            best["border_ratio"]
        ])

    # 保存统计信息
    rank_meta_csv = os.path.join(
        OUTPUT_ROOT,
        f"sam_predictor_meta_rank{args.rank}.csv"
    )

    with open(rank_meta_csv, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "image_path",
            "status",
            "num_sam_masks",
            "num_candidates",
            "best_score",
            "area_ratio",
            "center_overlap",
            "border_ratio"
        ])

        writer.writerows(meta_rows)

    print("\nFinished!")
    print("Mask output:", MASK_ROOT)
    print("Vis output:", VIS_ROOT)
    print("Food RGBA output:", FOOD_RGBA_ROOT)
    print("Meta CSV:", rank_meta_csv)


if __name__ == "__main__":
    main()