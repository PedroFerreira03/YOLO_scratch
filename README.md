# YOLO Object Detection with Custom Loss and mAP Evaluation

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

## 🧱 Project Structure

```bash
├── model/                  # YOLO model definition
├── utils/                  # IOU, loss, mAP, and other helper functions
├── train/                  # Training and evaluation scripts
├── data/                   # Dataloader and preprocessing (not shown here)
├── notebooks/
│   └── yolo_training.ipynb # Jupyter notebook with full training pipeline
├── best_model.pth          # (Optional) Entire model object
├── best_model_state.pth    # Best state dict saved during training
└── README.md               # You're reading it!
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/yolo-object-detection.git
cd yolo-object-detection
```

### 2. Install dependencies

```bash
pip install torch torchvision matplotlib
```

### 3. Prepare dataset

- Format: Each image should have labels in the format `(class_idx, x_center, y_center, width, height)`.
- Labels must be normalized in range [0, 1].

*(You may adapt your dataset or use Tiny ImageNet with bounding boxes.)*

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
- Classification loss (cross-entropy via MSE on one-hot)

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

## 🛠️ To Do

- [ ] Add data augmentation support
- [ ] Add support for COCO-style annotations
- [ ] TensorBoard integration
- [ ] Convert to ONNX for deployment

---

## 🤝 Acknowledgments

- Inspired by the YOLO (You Only Look Once) family of object detectors.
- Built using PyTorch for academic/research use.

---

## 📜 License

This project is licensed under the MIT License.

---

## ✉️ Contact

For questions or collaborations, feel free to open an issue or reach out.


