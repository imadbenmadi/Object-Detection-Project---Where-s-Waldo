# Object Detection Project - Where's Waldo?

## Project Overview

This project implements a simplified object detection system to identify and locate cartoon characters from the "Where's Waldo?" series. The system uses both a custom-built neural network architecture and a fine-tuned YOLOv8 model, comparing their performance on a synthetic dataset.

## Dataset Creation

### Character Selection

We selected three main characters from the "Where's Waldo?" series:

-   **Waldo**: Characterized by his iconic red and white striped shirt and hat
-   **Wilma**: Waldo's friend with blue clothing
-   **Wenda**: Character with pink/red striped clothing

### Background Collection

-   Used `icrawler` library to download 100+ themed background images
-   Backgrounds include cartoon scenes, crowded illustrations, and Where's Waldo-style puzzles
-   Images were resized to 640×640 pixels for consistency

### Synthetic Dataset Generation Process

1. **Object Preparation**:

    - Characters were extracted with transparent backgrounds using image editing tools
    - Each character was saved as a PNG with alpha channel

2. **Dataset Generation**:

    - Generated 5,000 training images, 1,000 validation images, and 200 test images
    - For each image:
        - Randomly selected a background
        - Randomly chose one character
        - Randomly scaled the character (50-100% of original size)
        - Placed character at random coordinates on the background
        - Calculated bounding box coordinates in YOLO format: `<class_id> <x_center> <y_center> <width> <height>`
        - Saved the image and corresponding annotation

3. **Data Format**:
    - Images saved as JPG files
    - Annotations saved in YOLO format (normalized coordinates)
    - Directory structure compatible with both custom training and YOLOv8

## Custom Model Architecture

### Backbone Network

-   **ResNet18**: Pre-trained on ImageNet, used as feature extractor
-   Removed final fully connected layer to obtain feature maps
-   Extracted features from multiple layers for multi-scale detection

### Feature Pyramid Network (FPN)

-   Implemented a simplified FPN to merge features from different scales
-   Used 1×1 convolutions to reduce channel dimensions
-   Added skip connections between layers for better gradient flow

### Detection Heads

1. **Classification Head**:

    - Two fully connected layers (512→256→3)
    - Outputs class probabilities for the 3 characters
    - Softmax activation for final prediction

2. **Regression Head**:
    - Two fully connected layers (512→256→4)
    - Outputs normalized coordinates (x_center, y_center, width, height)
    - No activation function on output layer

### Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [1, 64, 320, 320]           9,408
       BatchNorm2d-2         [1, 64, 320, 320]             128
              ReLU-3         [1, 64, 320, 320]               0
         MaxPool2d-4         [1, 64, 160, 160]               0
        BasicBlock-5         [1, 64, 160, 160]          73,984
        BasicBlock-6         [1, 64, 160, 160]          73,984
        BasicBlock-7        [1, 128, 80, 80]          230,144
        BasicBlock-8        [1, 128, 80, 80]          295,424
        BasicBlock-9        [1, 256, 40, 40]          919,296
       BasicBlock-10        [1, 256, 40, 40]        1,180,672
       BasicBlock-11        [1, 512, 20, 20]        3,675,648
       BasicBlock-12        [1, 512, 20, 20]        4,723,712
         AdaptiveAvgPool2d-13            [1, 512, 1, 1]               0
                  FPN-14            [1, 256, 20, 20]          45,056
           SPP-Module-15                   [1, 256]         393,472
             Linear-16                   [1, 256]         131,328
             Linear-17                     [1, 3]             771
             Linear-18                   [1, 256]         131,328
             Linear-19                     [1, 4]           1,028
================================================================
Total params: 11,885,383
Trainable params: 11,885,383
Non-trainable params: 0
```

## Training Process

### Data Augmentation

-   **Horizontal flips**: 50% probability
-   **Color jitter**: Randomized brightness (±0.2), contrast (±0.2), saturation (±0.2), hue (±0.1)
-   **Normalization**: Using ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]

### Loss Functions

-   **Classification Loss**: Cross-Entropy Loss
-   **Regression Loss**: Smooth L1 Loss (Huber Loss)
-   **Total Loss**: Classification Loss + λ \* Regression Loss (λ=10)

### Training Parameters

-   **Optimizer**: Adam with learning rate of 0.001
-   **Weight decay**: 1e-4 for regularization
-   **Batch size**: 32
-   **Epochs**: 50 with early stopping
-   **Learning rate scheduler**: ReduceLROnPlateau (patience=5, factor=0.1)
-   **Device**: CUDA GPU (when available)

### Training Loop

The training loop included:

1. Forward pass through the model
2. Computation of classification and regression losses
3. Backpropagation of gradients
4. Optimizer step
5. Learning rate scheduling
6. Validation after each epoch
7. Model checkpoint saving (best model based on validation loss)
8. Early stopping if no improvement for 10 epochs

## YOLOv8 Implementation

### Model Selection

-   Used the YOLOv8 nano model (`yolov8n.pt`)
-   Lightweight model with 3.2M parameters for efficient training and inference

### Fine-tuning Process

-   Created YAML configuration file with dataset paths and class names
-   Used Ultralytics API for fine-tuning:

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='dataset.yaml', epochs=50, imgsz=640, batch=16)
```

### YOLOv8 Training Parameters

-   **Learning rate**: 0.01 with cosine scheduler
-   **Optimizer**: SGD with momentum
-   **Image size**: 640×640
-   **Batch size**: 16
-   **Epochs**: 50
-   **Augmentation**: Default YOLOv8 augmentations (mosaic, mixup, etc.)

## Evaluation Metrics

### Performance Metrics

Both models were evaluated using:

1. **Mean Average Precision (mAP@0.5)**: Primary metric for object detection
2. **Intersection over Union (IoU)**: Measures overlap between predicted and ground truth boxes
3. **Classification Accuracy**: Percentage of correctly classified objects
4. **Precision**: TP / (TP + FP)
5. **Recall**: TP / (TP + FN)
6. **F1-Score**: Harmonic mean of precision and recall

### Results Visualization

-   Loss curves (training and validation)
-   Prediction visualization on test images
-   Confusion matrices for classification accuracy
-   Precision-Recall curves
-   mAP calculation at different IoU thresholds

## Implementation Details

### Key Libraries Used

-   **PyTorch**: Main deep learning framework
-   **torchvision**: For pre-trained models and transforms
-   **NumPy**: For numerical operations
-   **Matplotlib/Seaborn**: For visualization
-   **Pillow (PIL)**: Image processing
-   **tqdm**: Progress bars
-   **Ultralytics**: YOLO implementation
-   **icrawler**: Web scraping for background images

### Data Loading

-   Custom `SyntheticDataset` class inheriting from `torch.utils.data.Dataset`
-   Implemented `__getitem__` and `__len__` methods
-   Used `DataLoader` with shuffling, batching, and multi-processing

### Inference Pipeline

1. Load image and preprocess (resize, normalize)
2. Forward pass through model
3. Extract classification probabilities and bounding box coordinates
4. Apply confidence threshold (0.5)
5. Convert normalized coordinates to pixel coordinates
6. Draw bounding boxes and class labels on image

### Early Stopping Implementation

```python
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path)
```

## Challenges and Solutions

1. **Challenge**: Bounding box regression accuracy

    - **Solution**: Implemented Smooth L1 Loss and increased its weight in total loss

2. **Challenge**: Model overfitting due to simple backgrounds

    - **Solution**: Added more complex backgrounds and increased data augmentation

3. **Challenge**: Balancing classification and regression tasks

    - **Solution**: Tuned the λ parameter to balance the two loss components

4. **Challenge**: Small objects detection
    - **Solution**: Implemented Feature Pyramid Network to enhance multi-scale capabilities

## Future Improvements

1. Extend to multi-object detection with Non-Maximum Suppression
2. Implement more advanced backbones (EfficientNet, Vision Transformers)
3. Add more background diversity for better generalization
4. Test on real Where's Waldo puzzle images
5. Implement anchor-based detection for better accuracy
6. Add attention mechanisms to focus on important features

#### **production-ready interface where the user can test the mdoel**

A tkinter-based GUI application that allows users to select background images and character objects (like Waldo), place them via drag-and-drop, and simulate object detection with visualization of results.


[Watch the video](https://youtu.be/1VliO8UtMd8)


![Image Description](https://github.com/user-attachments/assets/d2b3b05c-bea3-4efc-aa68-64450ba276d7)
<video controls width="600">

  <source src="https://github.com/user-attachments/assets/339f8765-a359-41c3-ae68-3f9cea109be9" type="video/mp4">
  Your browser does not support the video tag.
</video>


## Conclusion

This project successfully demonstrates the implementation of a simplified object detection system using both custom neural networks and state-of-the-art YOLOv8 models. The synthetic dataset approach provides a controlled environment for training and evaluation, while the comparison between models offers insights into different object detection paradigms.

Similar code found with 3 license types
