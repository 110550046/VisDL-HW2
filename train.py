import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

from utils import DigitDataset, get_transform, collate_fn


def get_model(num_classes):
    """獲取預訓練的Faster R-CNN模型"""
    # 加載預訓練的Faster R-CNN模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # 替換分類器頭部以匹配我們的類別數
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_one_epoch(model, optimizer, data_loader, device):
    """訓練一個epoch"""
    model.train()
    total_loss = 0

    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    """評估模型性能"""
    model.eval()

    # 準備COCO格式的預測結果
    predictions = []
    targets_all = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = target["image_id"].item()
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    predictions.append(
                        {
                            "image_id": image_id,
                            "bbox": [
                                float(box[0]),
                                float(box[1]),
                                float(box[2] - box[0]),
                                float(box[3] - box[1]),
                            ],
                            "score": float(score),
                            "category_id": int(label),
                        }
                    )
                targets_all.append(target)

    # 創建COCO對象
    coco_gt = COCO()
    coco_gt.dataset = {
        "images": [{"id": i} for i in range(len(data_loader.dataset))],
        "annotations": [],
        "categories": [{"id": i} for i in range(1, 11)],
    }
    coco_gt.createIndex()

    # 創建預測結果
    coco_dt = coco_gt.loadRes(predictions)

    # 評估
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 返回mAP@[0.5:0.95]和mAP@0.5
    return coco_eval.stats[0], coco_eval.stats[1]


def main():
    # 設置設備
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 數據集路徑
    data_dir = "nycu-hw2-data"
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    train_json = os.path.join(data_dir, "train.json")
    valid_json = os.path.join(data_dir, "valid.json")

    # 創建訓練和驗證數據集
    train_dataset = DigitDataset(
        root_dir=train_dir, json_file=train_json, transform=get_transform(train=True)
    )

    valid_dataset = DigitDataset(
        root_dir=valid_dir, json_file=valid_json, transform=get_transform(train=False)
    )

    # 創建數據加載器
    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    # 創建模型
    num_classes = 11  # 10個數字類別 + 1個背景類
    model = get_model(num_classes)
    model.to(device)

    # 優化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 訓練循環
    num_epochs = 10
    best_map = 0.0

    for epoch in range(num_epochs):
        # 訓練一個epoch
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        # 每2個epoch進行一次驗證
        if (epoch + 1) % 2 == 0:
            map_50_95, map_50 = evaluate(model, valid_loader, device)
            print(f"Validation mAP@[0.5:0.95]: {map_50_95:.4f}, mAP@0.5: {map_50:.4f}")

            # 保存最佳模型
            if map_50 > best_map:
                best_map = map_50
                torch.save(model.state_dict(), "best_model.pth")
                print(f"保存最佳模型，mAP@0.5: {map_50:.4f}")

        # 每5個epoch保存一次檢查點
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    main()
