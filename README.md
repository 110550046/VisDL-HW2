# NYCU CV 2025 Spring HW2

```
StudentID:110550046
Name:吳孟謙
```

## Introduction

本專案實現了一個基於 Faster R-CNN 的數字檢測系統，用於識別圖像中的數字。該系統具有以下特點：

1. **模型架構**：使用 Faster R-CNN 與 ResNet-50-FPN 骨幹網絡。

2. **數據處理**：
   - 支持 COCO 格式的數據集
   - 實現了完整的數據增強流程，包括顏色抖動、大小調整等
   - 使用自定義的數據加載器和批處理函數

3. **訓練流程**：
   - 使用 SGD 優化器進行模型訓練
   - 實現了完整的訓練和驗證循環
   - 支持模型檢查點保存和最佳模型選擇

4. **推理功能**：
   - 支持單張圖片的數字檢測
   - 可以輸出 COCO 格式的檢測結果
   - 提供數字序列的預測功能

5. **評估指標**：
   - 使用 COCO 評估指標
   - 支持 mAP@[0.5:0.95] 和 mAP@0.5 的評估

## How to install & run

### 安裝

```bash
pip install -r requirements.txt
```

### 訓練模型

```bash
python train.py
```

### 推理

```bash
python inference.py
```

## Performance snapshot