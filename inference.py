import os
import json
import torch
import pandas as pd
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


def get_model(num_classes, model_path):
    """載入訓練好的模型"""
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    return model


def predict_image(model, image_path, device, score_threshold=0.5):
    """對單張圖片進行預測"""
    # 載入和預處理圖像
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image)

    # 進行預測
    model.eval()
    with torch.no_grad():
        prediction = model([image_tensor.to(device)])

    # 獲取預測結果
    boxes = prediction[0]["boxes"].cpu().numpy()
    scores = prediction[0]["scores"].cpu().numpy()
    labels = prediction[0]["labels"].cpu().numpy()

    # 過濾低置信度的預測
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    return boxes, scores, labels


def process_predictions(boxes, scores, labels):
    """處理預測結果，生成數字字符串"""
    if len(boxes) == 0:
        return -1

    # 根據x坐標排序，從左到右
    sorted_indices = boxes[:, 0].argsort()
    sorted_labels = labels[sorted_indices]

    # 將標籤轉換為數字字符串
    number = "".join(str(label) for label in sorted_labels)
    return int(number)


def main():
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入模型
    num_classes = 11  # 10個數字類別 + 1個背景類
    model_path = "model_epoch_10.pth"  # 使用最後一個epoch的模型
    model = get_model(num_classes, model_path)
    model.to(device)

    # 測試數據路徑
    test_dir = "nycu-hw2-data/test"

    # 準備結果存儲
    predictions_json = []
    predictions_csv = []

    # 處理每張測試圖片
    for image_name in os.listdir(test_dir):
        if not image_name.endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(test_dir, image_name)
        image_id = int(os.path.splitext(image_name)[0])

        # 預測
        boxes, scores, labels = predict_image(model, image_path, device)

        # 生成COCO格式的預測結果
        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min

            predictions_json.append(
                {
                    "image_id": image_id,
                    "bbox": [float(x_min), float(y_min), float(width), float(height)],
                    "score": float(score),
                    "category_id": int(label),
                }
            )

        # 生成數字預測結果
        pred_label = process_predictions(boxes, scores, labels)
        predictions_csv.append({"image_id": image_id, "pred_label": pred_label})

    # 保存預測結果
    with open("pred.json", "w") as f:
        json.dump(predictions_json, f)

    df = pd.DataFrame(predictions_csv)
    df.to_csv("pred.csv", index=False)


if __name__ == "__main__":
    main()
