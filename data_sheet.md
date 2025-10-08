# Datasheet Template - CIFAR-10 Dataset

## Motivation

**For what purpose was the dataset created?**

The CIFAR-10 dataset was created to serve as a benchmark for image classification algorithms, particularly for evaluating machine learning models on object recognition tasks. It was designed to be a subset of the larger 80 million tiny images dataset, providing a manageable yet challenging dataset for research and education in computer vision and machine learning.

**Who created the dataset and on behalf of which entity?**

The dataset was created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton at the Canadian Institute for Advanced Research (CIFAR). The dataset was developed as part of academic research at the University of Toronto. The creation was supported by CIFAR, NSERC (Natural Sciences and Engineering Research Council of Canada), and other academic funding sources.

**Who funded the creation of the dataset?**

The dataset creation was funded by the Canadian Institute for Advanced Research (CIFAR) and supported by academic grants from NSERC and the University of Toronto.

## Composition

**What do the instances that comprise the dataset represent?**

The instances in CIFAR-10 are 32×32 pixel color images (RGB) representing common images. Each image belongs to one of 10 mutually exclusive classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

**How many instances of each type are there?**

- **Total instances:** 60,000 images
- **Training set:** 50,000 images (5,000 per class)
- **Test set:** 10,000 images (1,000 per class)
- **Distribution:** Perfectly balanced - each class has exactly 6,000 images (5,000 training + 1,000 test)

**Is there any missing data?**

No. 

**Does the dataset contain data that might be considered confidential?**

No. The images were collected from public sources and do not raise confidentiality concerns.

## Collection Process

**How was the data acquired?**

The CIFAR-10 images were extracted from the 80 million tiny images dataset, which was collected by downloading images from the internet using image search engines. The collection process involved:

1. Querying search engines with noun keywords
2. Downloading returned images
3. Downsampling images to 32×32 pixels
4. Manually verifying and labeling a subset for the 10 chosen categories

**If the data is a sample of a larger subset, what was the sampling strategy?**

CIFAR-10 is a curated subset of the 80 million tiny images dataset. The sampling strategy was:

- **Category selection:** 10 mutually exclusive categories were chosen to represent common, easily distinguishable objects
- **Balanced sampling:** Exactly 6,000 images per category were selected to ensure class balance
- **Quality filtering:** Images were manually reviewed to ensure they clearly represented their assigned category
- **Random split:** The dataset was randomly split into 50,000 training and 10,000 test images while maintaining class balance

**Over what time frame was the data collected?**

The CIFAR-10 dataset was created and published in 2009 by researchers from University of Toronto. The original 80 million tiny images dataset (from which CIFAR-10 was derived) was collected between 2006-2008 through automated web scraping and search engine queries by researchers from MIT.

## Preprocessing/Cleaning/Labelling

**Was any preprocessing/cleaning/labeling of the data done?**

Yes, significant preprocessing and labeling was performed:

**Preprocessing:**
- **Resizing:** All images were downsampled to a uniform 32×32 pixel resolution
- **Format standardization:** Images were converted to RGB color format (3 channels)
- **Normalization:** Pixel values are stored as integers in the range [0, 255]

**Cleaning:**
- Manual inspection to remove mislabeled or ambiguous images
- Removal of images that didn't clearly represent their assigned category
- Elimination of duplicate or near-duplicate images
- Quality filtering to ensure images were recognizable despite low resolution

**Labeling:**
- Each image was assigned a single label from the 10 categories
- Labels were verified through manual review
- Ambiguous cases were resolved by consensus or removed from the dataset


**Was the "raw" data saved in addition to the preprocessed data?**

The raw, higher-resolution images from the 80 million tiny images dataset still exist, but CIFAR-10 is distributed only in its preprocessed 32×32 format. 

## Uses

**What other tasks could the dataset be used for?**

CIFAR-10 can be used for various machine learning and computer vision tasks:

- **Transfer learning:** Pre-training models before fine-tuning on other datasets
- **Educational purposes:** Teaching machine learning concepts, CNNs, and image classification
- **Algorithm benchmarking:** Comparing performance of different architectures and optimization techniques
- **AutoML research:** Hyperparameter optimization and neural architecture search (as in this project)
- **Continual learning:** Testing models' ability to learn classes sequentially

**Potential issues and considerations:**

**Limitations:**
- **Low resolution (32×32):** May not generalize well to high-resolution image tasks
- **Limited diversity:** Only 10 classes; real-world applications typically require more categories
- **Distribution shift:** Images collected from internet searches may not represent real-world image distributions
- **Geographic bias:** Likely overrepresents Western/North American contexts due to search engine biases

**Potential biases to be aware of:**
- Image quality and composition reflect internet search results from the late 2000s

**Mitigation strategies:**
- Do not assume models trained on CIFAR-10 will generalize to high-resolution images without fine-tuning
- Be aware of class imbalance when deploying in real-world scenarios (CIFAR-10 is artificially balanced)
- Use CIFAR-10 primarily for prototyping, education, and benchmarking rather than direct production deployment

**Are there tasks for which the dataset should not be used?**

CIFAR-10 should NOT be used for:

**Any high-stakes decision making process** without additional validation on domain-specific data. These data looks good for educational purposes.


**Ethical considerations:**
- Models trained solely on CIFAR-10 should not be deployed in contexts where errors could cause harm
- The limited class set means many real-world objects will be misclassified
- The low resolution may lead to over-reliance on texture rather than shape, which may not generalize

## Distribution

**How has the dataset already been distributed?**

CIFAR-10 is widely distributed and freely accessible through multiple channels:

**Official Sources:**
- **Primary website:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Technical report:** https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

**Direct downloads:**
- Python version: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz (163 MB)
- MATLAB version: https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz (175 MB)
- Binary version: https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz (162 MB)

**Programmatic access:**
- PyTorch: `torchvision.datasets.CIFAR10` (auto-downloads from official source)
- TensorFlow: `tensorflow.keras.datasets.cifar10`
- Keras: `keras.datasets.cifar10`

**Community resources:**
- Kaggle: https://www.kaggle.com/c/cifar-10
- Papers With Code: https://paperswithcode.com/dataset/cifar-10

The dataset is automatically downloaded by most machine learning libraries when first accessed, making it extremely accessible to researchers and practitioners worldwide.

**Is it subject to any copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?**

CIFAR-10 is released under a permissive license:

- **License:** The dataset is freely available for research and educational purposes
- **Copyright:** The dataset creators retain copyright but grant broad usage rights
- **No commercial restrictions:** Can be used in commercial research and applications
- **No registration required:** Freely accessible without account creation or formal agreements
- **Attribution:** Users are encouraged (but not legally required) to cite the original paper:
  - Krizhevsky, A. (2009). "Learning Multiple Layers of Features from Tiny Images" (Master's thesis)

## Maintenance

**Who maintains the dataset?**

**Primary maintainers:**
- **Alex Krizhevsky** (original creator) - Now at Google
- **Geoffrey Hinton** - University of Toronto and Google
- **University of Toronto** - Computer Science Department hosts the official dataset

**Maintenance status:**
- **Static dataset:** CIFAR-10 is considered a frozen benchmark dataset and has not been updated since its 2009 release
- **Website hosting:** The University of Toronto maintains the official website and download links
- **No active updates planned:** The dataset is intentionally kept unchanged to maintain consistency for benchmarking purposes
- **Community support:** The broader machine learning community maintains integration with popular frameworks (PyTorch, TensorFlow, etc.)

**Contact information:**
- Official website: https://www.cs.toronto.edu/~kriz/cifar.html
- For issues or questions: Contact through University of Toronto Computer Science Department

**Long-term preservation:**
- The dataset is mirrored on multiple servers worldwide
- Integrated into major ML frameworks ensures long-term availability
- Academic importance ensures continued hosting and accessibility
- The small size (163 MB) makes long-term storage trivial

**Version control:**
- **Current version:** CIFAR-10 (original and only version)
- **No versioning system:** The dataset is frozen and will not receive updates
- **Related datasets:** 
  - CIFAR-100 (same images, 100 fine-grained classes) maintained similarly
  - These are separate datasets, not versions

**Future maintenance:**
- Expected to remain available indefinitely due to widespread use in ML research and education
- No plans for updates or corrections (by design - benchmark stability)
- Community maintains integrations with modern tools and frameworks

