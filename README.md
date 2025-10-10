# AutoML Image Classifier: Bayesian-Optimized CIFAR-10 Recognition System


This project is about automating how to teach a computer to recognize everyday objects like cats, dogs, cars, and airplanes, similar to how children learn to identify things. This project builds an intelligent system that not only learns to classify images but also figures out the best way to learn by itself. Instead of manually adjusting hundreds of settings through trial and error (which could take weeks), the system uses a smart technique called Bayesian optimization to automatically find the optimal learning strategy in just a few hours. The result is a deployed web application where anyone can upload a photo and instantly get predictions about what object it contains. This demonstrates how artificial intelligence can automate complex decision-making processes, making machine learning more accessible and efficient for real-world applications.


## DATA

**Dataset:** CIFAR-10 (Canadian Institute for Advanced Research, 10 classes)

**Source:** 
- Official website: https://www.cs.toronto.edu/~kriz/cifar.html
- Accessed programmatically via PyTorch's `torchvision.datasets.CIFAR10`
- Direct download: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

**Dataset Description:**
- **Size:** 60,000 images total
  - Training set: 50,000 images
  - Test set: 10,000 images
- **Image specifications:** 32×32 pixels, RGB color (3 channels)
- **Classes (10):** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Distribution:** Perfectly balanced - 6,000 images per class
- **File size:** 163 MB (compressed)

For more details see the [data sheet](data_sheet.md).

## MODEL 
**Model Architecture:** SimpleCNN (Custom Convolutional Neural Network).

**Why SimpleCNN ?**
- Fast to train.
- Can be trained on common laptops hence no need to rent out massive GPUs.
- Easy to understand and explain
- Sufficiency. 85.6% accuracy is good enough for the 32×32 images
- Focus. Project is about understanding the end to end ML engineering. That is, how models are developed and deployed to production for day to day use.

See [model card](model_card.md) for more details.

## HYPERPARAMETER OPTIMSATION

**The Challenge:** There are millions of possible combinations of settings to get the best results from the model. Testing them all to find the optimimum settigns manually would take years.

**The Solution:** We used **Optuna**, a smart tool that uses **Bayesian optimization** to automatically find the best settings. Instead of trying random combinations, Optuna learns from each attempt and gets smarter about what to try next.

### The 5 Settings We Optimized

| Hyperparameter | Search Range / Options | Rationale |
| :--- | :--- | :--- |
| **Learning Rate** | 0.00001 to 0.1 (log scale) | Controls the speed of learning. |
| **Batch Size** | [32, 64, 128, 256] | Balances training speed and generalization. |
| **Optimizer** | [Adam, SGD, AdamW] | The algorithm for updating the model's weights. |
| **Weight Decay** | 0.000001 to 0.01 (log scale) | A regularization technique to prevent overfitting. |
| **Dropout Rate** |0.0 to 0.6 | Randomly deactivates neurons to build a robust model. |

## RESULTS

**Performance Summary:**
- Baseline (manual settings with 30 epoch): 82.77%
- Optuna optimization(30 epoch): 82.88%  
- Final model using optimal hyperparameters (75 epoch): **85.60%**
- **Total improvement: +2.83%**

### Key Findings

Bayesian optimization found slightly better hyperparameters (+0.11% improvement) in 3 hours and 31 minutes, but the real breakthrough came from training longer. By doubling the training time from 30 to 75 epochs, we gained an additional 2.72% accuracy.

**Best Settings Discovered:**
- Learning rate: 0.000315 (conservative, careful learning)
- Batch size: 32 (smallest option - noisier but better generalization)
- Optimizer: AdamW (superior to Adam and SGD)
- Dropout: 42.5% (high regularization prevented overfitting)
- Weight decay: 0.000254 (minimal but effective)

### What We Learned

The optimization successfully automated hyperparameter search, saving weeks of manual experimentation. However, the biggest lesson was unexpected: **training patience matters more than perfect settings**. The model needed time to fully learn patterns in the data.

This demonstrates that AutoML isn't magic - it's a valuable tool that finds good configurations quickly, but proper training duration remains essential for achieving strong results.

## CONTACT DETAILS

**Author:** Justus Mbaluka 
**Project:** ML Engineering Portfolio - AutoML Image Classifier  
**Institution:** Imperial College London: Fundamental AI and ML Programme.

**Connect with me:**
- 💼 **LinkedIn:** [linkedin.com/in/jkyalo](https://linkedin.com/in/jkyalo)
- 💻 **GitHub:** [@kyaloj](https://github.com/kyaloj)


**For project inquiries or collaboration opportunities, please connect via LinkedIn.**

