# YOLO Object Detection with Custom Loss and mAP Evaluation

`NOTE`: I messed up when changing the work environment, which ended up deletin the 30+ commits that I had done over the work in the project.

This repository contains an end-to-end implementation of a YOLO-style object detection model using PyTorch. The project includes custom loss computation, mean Average Precision (mAP) evaluation, and full training/validation routines, designed with flexibility for datasets like Tiny ImageNet or any dataset with bounding box annotations.

---

## 📌 Features

- ✅ YOLO-style bounding box prediction with multiple anchors per cell.
- ✅ Custom `YOLOLoss` implementation with support for coordinate loss, confidence loss, and classification loss.
- ✅ IoU-based responsibility assignment for predicted boxes.
- ✅ Full `mAP` evaluation pipeline with class-wise precision-recall computation.
- ✅ Training loop with early stopping and best model saving.
- ✅ Modular design and GPU support.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/PedroFerreira03/YOLO_scratch
cd YOLO_scratch
```

### 2. Install dependencies

```bash
pip install torch torchvision matplotlib
```

### 3. Prepare dataset
- This project was done using *tiny-imagenet*. You may need a license to use the dataset.

---

## 🏗️ Model Overview

The model predicts:

- For each grid cell:
  - **B bounding boxes**: Each with `(confidence, x, y, w, h)`
  - **C class probabilities**

Total output shape: `(Batch_size, S, S, 5 * B + C)`

---

## 🔍 Loss Function: `YOLOLoss`

Implements:

- Coordinate loss (x, y, w, h)
- Confidence loss for object/no-object
- Classification loss

Weighting parameters:
- `l_coord`: weight for localization (default: `5`)
- `l_noobj`: weight for no-object confidence (default: `0.5`)

---

## 📏 Evaluation: mAP

The project includes full `mAP` computation:

- Per-class precision-recall at different IoU thresholds
- Supports arbitrary number of classes
- Average Precision (AP) is calculated using 11-point interpolation

```python
mAP(predicted, ground_truth, iou_thresholds=[0.5], num_classes=200)
```

---

## 🏋️ Training & Evaluation

```python
model = YOLO(C=num_classes).to(device)
criterion = YOLOLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
    val_loss, mAP = evaluate(model, val_loader, criterion, device)

    # Early stopping logic...
```

Model saving:
```python
torch.save(best_model_state, "best_model_state.pth")
torch.save(best_model, "best_model.pth")  # optional full model
```

---

## 📊 Example Output

During training:

```
Beginning Training on epoch 0
0.00% done
10.00% done
...
Train Loss at epoch 0: 9.421
Val Loss at epoch 0: 10.207
mAP at epoch 0: 0.427
```

---

## 🤝 Acknowledgments

- Inspired by the YOLO (You Only Look Once) family of object detectors.
- Built using PyTorch for academic/research use.
- Used a small subset of *imagenet* dataset.

---

## 📜 License

This project is licensed under the MIT License.

---

## ✉️ Contact

For questions or collaborations, feel free to open an issue or reach out.


