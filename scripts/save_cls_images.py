# -*- coding: utf-8 -*-
import os
import glob
import cv2
import numpy as np

# 这里可以放中文路径/中文文件夹名
IMG_DIR = r"E:\images\food\gen_result\images"   # 图像文件夹
LAB_DIR = r"E:\images\food\gen_result\labels"   # 标签文件夹
OUT_DIR = r"E:\images\food\gen_result\images_cls"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

start_num = 0 ##所有数字类别编号+200

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def find_image_by_stem(images_dir, stem):
    for ext in IMG_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    cand = glob.glob(os.path.join(images_dir, stem + ".*"))
    for p in cand:
        if os.path.splitext(p)[1].lower() in IMG_EXTS:
            return p
    return None


def yolo_to_xyxy(cx, cy, w, h, W, H):
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H

    # clamp
    x1 = max(0, min(W - 1, int(round(x1))))
    y1 = max(0, min(H - 1, int(round(y1))))
    x2 = max(0, min(W,     int(round(x2))))
    y2 = max(0, min(H,     int(round(y2))))
    return x1, y1, x2, y2


def unique_path(path):
    """如果已存在则追加 _dup001 之类，避免覆盖"""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    k = 1
    while True:
        p2 = f"{base}_dup{k:03d}{ext}"
        if not os.path.exists(p2):
            return p2
        k += 1

# 支持中文路径的读取函数
def cv_imread(file_path):
    # 读取中文路径图片
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img

# 支持中文路径的保存函数
def cv_imwrite(file_path, img, quality=95):
    # 保存到中文路径
    cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tofile(file_path)


def process_all():
    ensure_dir(OUT_DIR)

    label_files = sorted(glob.glob(os.path.join(LAB_DIR, "*.txt")))
    if not label_files:
        print(f"no label files in: {LAB_DIR}")
        return

    saved, missing_imgs, bad_rois = 0, 0, 0

    for lf in label_files:
        stem = os.path.splitext(os.path.basename(lf))[0]
        img_path = find_image_by_stem(IMG_DIR, stem)
        if img_path is None:
            missing_imgs += 1
            continue

        # 使用支持中文的读取函数
        img = cv_imread(img_path)
        if img is None:
            missing_imgs += 1
            continue
        H, W = img.shape[:2]

        with open(lf, "r", encoding="utf-8") as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]
        if not lines:
            continue

        # ROI 序号用“该标注在文件中的行序号”
        for roi_idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                cx = float(parts[1]); cy = float(parts[2])
                bw = float(parts[3]); bh = float(parts[4])
            except ValueError:
                continue

            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, W, H)
            if x2 <= x1 or y2 <= y1:
                bad_rois += 1
                continue

            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                bad_rois += 1
                continue

            out_cls_dir = os.path.join(OUT_DIR, str(cls_id+start_num))
            ensure_dir(out_cls_dir)

            out_name = f"{stem}_{roi_idx:03d}.jpg"
            out_path = unique_path(os.path.join(out_cls_dir, out_name))

            # 使用支持中文的保存函数
            cv_imwrite(out_path, roi, 95)
            saved += 1

    print(f"saved={saved}, missing_imgs={missing_imgs}, bad_rois={bad_rois}")


if __name__ == "__main__":
    process_all()