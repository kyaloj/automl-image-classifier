# Model Card

See the [example Google model cards](https://modelcards.withgoogle.com/model-reports) for inspiration. 

## Model Description

**Input:** 

The model accepts RGB color images as input with the following specifications:
- **Format:** JPEG, PNG, or any standard image format
- **Color space:** RGB (3 channels)
- **Input size:** Any resolution (automatically resized to 32×32 pixels during preprocessing)

**Output:** 

The model produces multi-class classification predictions:
- **Primary output:** Single predicted class label from 10 categories
- **Confidence score:** Probability (0-100%) for the predicted class
- **Top-3 predictions:** Three most likely classes with their respective confidence scores
- **Output format:** JSON response containing:
```json
  {
    "predicted_class": "cat",
    "confidence": "87.45%",
    "top_3_predictions": [
      {"class": "cat", "confidence": "87.45%"},
      {"class": "dog", "confidence": "8.23%"},
      {"class": "deer", "confidence": "2.11%"}
    ],
  }
```

**Model Architecture:**

```

Input: 32×32×3 RGB image
├── Conv Block 1
│   ├── Conv2D: 3 → 32 filters, 3×3 kernel, padding=1
│   ├── ReLU activation
│   └── MaxPool2D: 2×2 (output: 16×16×32)
│
├── Conv Block 2
│   ├── Conv2D: 32 → 64 filters, 3×3 kernel, padding=1
│   ├── ReLU activation
│   └── MaxPool2D: 2×2 (output: 8×8×64)
│
├── Conv Block 3
│   ├── Conv2D: 64 → 128 filters, 3×3 kernel, padding=1
│   ├── ReLU activation
│   └── MaxPool2D: 2×2 (output: 4×4×128)
│
├── Flatten: 4×4×128 = 2,048 features
│
├── Fully Connected Block 1
│   ├── Linear: 2,048 → 512
│   ├── ReLU activation
│   └── Dropout: 0.32 (optimized rate)
│
├── Fully Connected Block 2
│   └── Linear: 512 → 10 (output classes)
│
└── Output: Softmax probabilities (applied during inference)

```

## Performance

Using a simple formulae of Total Correct Prediction (True Positives and True Negatives) / Total Predictions, we determine the model performance.

```
Baseline Best Accuracy: 82.77%

Best Hyperparameters during Optimzation:
  learning_rate: 0.00031489116479568613
  batch_size: 32
  optimizer: AdamW
  weight_decay: 0.00025378155082656634
  dropout: 0.4248435466776273

Optimization best accuracy: 82.88%
Final model test accuracy:  85.60%
```

Interestingly, we got those best Hyper params during the first optimization trial. It took about 3 hrs 30 mins in the optimization phase.


## Limitations

- Model only processes 32×32 pixel images.
- Only recognizes 10 specific object categories.
- Designed for images containing primarily one object.
- Trained exclusively on CIFAR-10 (internet images from 2006-2008) hence performance degrades on images with different conditions such as lighting and viewing angles.
- Model doesn't provide explanations for predictions.
- Requires ~3 to 4 hours for full training on consumer hardware.

## Trade-offs

**Model Complexity vs. Accuracy**

Chosen approach: Simple CNN

Advantages:

- Fast training (~30 minutes for 30 epochs on consumer hardware)
- Small model size (4.8 MB - easily deployable)
- Low inference latency
- Interpretable architecture
- Suitable for educational purposes and showcasing my career goals of AutoML projects.


Disadvantages:

- Lower accuracy (85.6%) compared to other online models such as ResNet-18 or larger models
- Limited capacity to learn complex features.

**Automated hyperparameter selection vs. manual expert tuning**
Chosen approach: Bayesian optimization with Optuna (AutoML)

Advantages:

- Removes human bias and guesswork
- Systematically explores search space
- Reproducible results
- Requires minimal ML expertise
- Can discover non-intuitive hyperparameter combinations


Disadvantages:

- Less intuitive than manual tuning for experts
- Requires defining search space upfront
- May miss domain-specific insights
- Cannot incorporate theoretical knowledge easily