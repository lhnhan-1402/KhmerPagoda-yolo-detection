from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np
import torch
from torchvision.ops import nms

model = YOLO(r"D:\IT\CT201e\runs\detect\yolov5m_epoch_250\weights\best.pt")


def center_distance(boxA, boxB):
    cxA = (boxA[0] + boxA[2]) / 2
    cyA = (boxA[1] + boxA[3]) / 2
    cxB = (boxB[0] + boxB[2]) / 2
    cyB = (boxB[1] + boxB[3]) / 2
    return np.sqrt((cxA - cxB)**2 + (cyA - cyB)**2)


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def predict_image(image_path):

    results = model.predict(source=image_path, conf=0.1, verbose=False)
    result = results[0]

    img_original = cv2.imread(image_path)
    img_improved = img_original.copy()

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    img_yolo = result.plot()

    # ===============================
    # ===== Confidence Boosting =====
    # ===============================

    DIST_THRESHOLD = 80
    AREA_RATIO_LIMIT = 3.0
    TOP_K = 3

    new_confs = confs.copy()
    unique_classes = np.unique(classes)

    for cls in unique_classes:

        indices = np.where(classes == cls)[0]
        if len(indices) == 0:
            continue

        # 🔥 Boost riêng cho class nhỏ (la)
        if model.names[int(cls)] == "la":
            BOOST_FACTOR = 0.35
        else:
            BOOST_FACTOR = 0.25

        sorted_idx = indices[np.argsort(confs[indices])]
        top_k_idx = sorted_idx[-TOP_K:]

        for anchor_idx in top_k_idx:

            anchor_box = boxes[anchor_idx]
            anchor_conf = confs[anchor_idx]
            anchor_area = box_area(anchor_box)

            for idx in indices:
                if idx == anchor_idx:
                    continue

                dist = center_distance(anchor_box, boxes[idx])
                area = box_area(boxes[idx])
                area_ratio = max(anchor_area, area) / (min(area, anchor_area) + 1e-6)

                if dist < DIST_THRESHOLD and area_ratio < AREA_RATIO_LIMIT:
                    boosted = new_confs[idx] + BOOST_FACTOR * anchor_conf
                    new_confs[idx] = min(boosted, 1.0)

    # ===============================
    # ===== Filter + NMS lại =====
    # ===============================

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(new_confs, dtype=torch.float32)
    classes_tensor = torch.tensor(classes, dtype=torch.int64)

    final_keep_indices = []

    for cls in classes_tensor.unique():

        cls_mask = classes_tensor == cls
        cls_boxes = boxes_tensor[cls_mask]
        cls_scores = scores_tensor[cls_mask]

        class_name = model.names[int(cls)]

        # 🔥 threshold riêng cho từng class
        if class_name == "la":
            CONF_THRESHOLD = 0.2
            IOU_THRESHOLD = 0.8   # 🔥 giữ nhiều box nhỏ
        else:
            CONF_THRESHOLD = 0.3
            IOU_THRESHOLD = 0.6

        # lọc confidence
        keep_mask = cls_scores > CONF_THRESHOLD
        cls_boxes = cls_boxes[keep_mask]
        cls_scores = cls_scores[keep_mask]

        if len(cls_boxes) == 0:
            continue

        keep = nms(cls_boxes, cls_scores, IOU_THRESHOLD)

        cls_indices = torch.where(cls_mask)[0][keep_mask]
        final_keep_indices.extend(cls_indices[keep].tolist())

    # ===============================
    # ===== Vẽ kết quả cuối =====
    # ===============================

    annotator = Annotator(img_improved)

    for idx in final_keep_indices:
        x1, y1, x2, y2 = boxes_tensor[idx]
        cls_id = int(classes_tensor[idx])
        conf = float(scores_tensor[idx])
        label = f"{model.names[cls_id]} {conf:.2f}"
        annotator.box_label([x1, y1, x2, y2], label)

    img_improved = annotator.result()

    cv2.imshow("Original YOLO", img_yolo)
    cv2.imshow("YOLO + Boost + NMS", img_improved)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


predict_image(r"D:\IT\KhmerPagoda-detection\Dataset\test\images\image-805-_JPG_JPG.rf.abd16f83faac018869029c91b919e9df.jpg")