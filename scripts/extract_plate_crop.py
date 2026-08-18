import os
import cv2


# ============================
# 路径配置
# ============================

DATA_ROOT = "/data/ljy/dish_detect/lichu_dish_cls"

IMAGE_DIR = os.path.join(DATA_ROOT, "images")
LABEL_DIR = os.path.join(DATA_ROOT, "labels")

OUTPUT_DIR = os.path.join(DATA_ROOT, "plate_crops")
VIS_DIR = os.path.join(DATA_ROOT, "vis")


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


# ============================
# YOLO bbox转换
# ============================

def yolo_to_xyxy(label, img_w, img_h):

    cls_id, cx, cy, bw, bh = label

    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)

    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)


    # 防止越界
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(img_w - 1, x2)
    y2 = min(img_h - 1, y2)

    return int(cls_id), x1, y1, x2, y2



# ============================
# 处理所有图片
# ============================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]


print("Total images:", len(image_files))


count = 0


for img_name in image_files:


    img_path = os.path.join(
        IMAGE_DIR,
        img_name
    )


    label_name = os.path.splitext(img_name)[0] + ".txt"

    label_path = os.path.join(
        LABEL_DIR,
        label_name
    )


    if not os.path.exists(label_path):
        print("No label:", img_name)
        continue


    img = cv2.imread(img_path)


    if img is None:
        print("Cannot read:", img_name)
        continue


    h, w = img.shape[:2]


    # 可视化
    vis_img = img.copy()


    # 读取label

    with open(label_path, "r") as f:
        lines = f.readlines()



    for idx, line in enumerate(lines):

        line=line.strip()

        if len(line)==0:
            continue


        values = list(map(float, line.split()))


        cls_id, x1, y1, x2, y2 = yolo_to_xyxy(
            values,
            w,
            h
        )


        # =====================
        # 裁剪盘子
        # =====================

        crop = img[
            y1:y2,
            x1:x2
        ]


        if crop.size == 0:
            continue



        # 创建类别文件夹

        class_dir = os.path.join(
            OUTPUT_DIR,
            f"plate_{cls_id}"
        )

        os.makedirs(
            class_dir,
            exist_ok=True
        )


        save_name = (
            os.path.splitext(img_name)[0]
            +
            f"_plate{cls_id}_{idx}.jpg"
        )


        save_path=os.path.join(
            class_dir,
            save_name
        )


        cv2.imwrite(
            save_path,
            crop
        )



        # =====================
        # 可视化
        # =====================

        cv2.rectangle(
            vis_img,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )


        cv2.putText(
            vis_img,
            f"plate_{cls_id}",
            (x1,y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )


        count += 1



    vis_path=os.path.join(
        VIS_DIR,
        img_name
    )

    cv2.imwrite(
        vis_path,
        vis_img
    )



print("\nFinished!")
print("Total plate crops:", count)

print("\nOutput:")
print(OUTPUT_DIR)
print(VIS_DIR)