# 🧠 Smart Fruit & Vegetable Classifier 🍎🥦

This project is a **fruit and vegetable image classification system** built using **Transfer Learning**. It classifies images into their correct categories using a trained deep learning model.



## 📁 Dataset

The dataset used contains images of various fruits and vegetables, divided into:
- **Training**
- **Validation**
- **Testing**

📦 Dataset Source: [Kaggle - Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)

---

## 🧠 Model Architecture

We used a pre-trained CNN model:
- **Base Model**: `MobileNetV2`
- **Fine-tuned** for 36 classes
- **Saved as**: `models/fruit_classifier.h5`

---

## 🔍 How to Use

1. Place your test image in the project folder.
2. Run `predict.py` with your image path.
3. The model will display the predicted class.

```bash
python predict.py --image sample.jpg

