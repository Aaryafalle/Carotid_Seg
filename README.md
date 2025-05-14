
# Carotid Ultrasound Image Segmentation using U-Net

This project performs automatic **segmentation of carotid artery regions** in ultrasound images using a **U-Net deep learning model**. The dataset contains real-world medical ultrasound images and their corresponding expert-annotated masks.

---

## 🧠 What is Carotid?

The **carotid arteries** are the **two major blood vessels in the neck** that supply blood to the brain, neck, and face. Detecting blockages or abnormalities in these arteries is crucial for **preventing strokes** and **monitoring vascular health**. 

Medical ultrasound imaging is commonly used to scan these arteries because it is **non-invasive**, **safe**, and **cost-effective**.

---

## 🎯 Project Goal

To build an accurate **semantic segmentation model** using U-Net that can automatically detect and segment the **carotid artery** in grayscale ultrasound images.

---

## 🧰 Technologies Used

- Python 🐍
- TensorFlow / Keras
- OpenCV & PIL
- NumPy, Pandas, Matplotlib
- Sklearn (train/test split)
- KaggleHub (to download dataset)

---

## 📁 Dataset Info

- Source: [Kaggle - Carotid Ultrasound Images](https://www.kaggle.com/datasets/orvile/carotid-ultrasound-images)
- Total Images: **1100**
- Each image has a **corresponding binary mask** (0 = background, 1 = carotid artery).
- Image format: `.jpg`, Mask format: `.png`

---

## 📦 Steps Performed

### 1. Dataset Loading & Validation
- Downloaded the dataset using `kagglehub`.
- Matched each image with its corresponding mask.
- Ensured no missing or duplicate files.

### 2. Preprocessing
- Resized images and masks to **256x256** pixels.
- Normalized pixel values to [0, 1] range.
- Split data into **80% training** and **20% testing** sets.

### 3. U-Net Model

#### 🔍 What is U-Net?
U-Net is a deep learning segmentation algorithm.
It is based on CNNs, designed specifically for biomedical image segmentation.
The name U-Net comes from its U-shaped architecture, which has:
A contracting path (encoder): learns context using convolution and pooling.
An expanding path (decoder): enables precise localization using upsampling and skip connections.

#### 📌 Why U-Net for this project?
~Carotid artery segmentation is a pixel-wise classification task.
~U-Net performs well on small datasets and gives accurate segmentation masks for medical images.

### 4. Training
- Loss Function: `binary_crossentropy`
- Optimizer: `Adam`
- Metrics: `accuracy`
- EarlyStopping & ReduceLROnPlateau used to improve training.
- Trained for 5 epochs.

### 5. Results
- Achieved **99.34% accuracy** on the test set.
- Model correctly segmented carotid arteries in unseen images.

---

## 📊 Visual Results

Examples of segmented outputs:

| Original Image | Ground Truth Mask | Predicted Mask |
|----------------|-------------------|----------------|
|     ✅ Img      |        ✅ Mask      |     ✅ Output   |

---

## 🧪 Future Improvements

- Use **Dice Coefficient** or **IoU** for better segmentation metrics.
- Train longer (more epochs) for enhanced performance.
- Use **data augmentation** to improve generalization.
- Try other models like **ResUNet**, **Attention U-Net**, or **TransUNet**.

---

## 🧑‍⚕️ Real-World Application

This model can be used in **clinical support tools** for radiologists or vascular surgeons, helping to:
- Identify blocked or narrowed carotid arteries.
- Monitor patients over time.
- Reduce diagnostic error.

---

## 🤝 Acknowledgements

- Dataset by [Orvile](https://www.kaggle.com/datasets/orvile/carotid-ultrasound-images)
- U-Net architecture based on the paper: *U-Net: Convolutional Networks for Biomedical Image Segmentation* by Olaf Ronneberger et al.

---

## 📌 Author

- ✍️ Developed by: *Aarya Falle*  
- 💡 Guided by: U-Net paper and community notebooks  
