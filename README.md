# Neural Style Transfer (NST) Using VGG19

## 📌 Project Overview
This project implements **Neural Style Transfer (NST)** using **TensorFlow and VGG19**. It applies the artistic style of one image (e.g., a painting) to another image while preserving its content.

## 🚀 Features
- Uses **VGG19** to extract content and style features.
- Applies **artistic styles** to photos.
- Supports **custom content & style images**.
- Uses **deep learning optimization** to blend images.

## ⚙️ Installation
### **1️⃣ Install Dependencies**
Ensure you have Python installed, then run:
```bash
pip install tensorflow numpy matplotlib pillow
```

## 📜 Usage
### **Run the script:**
```bash
python neural_style_transfer.py
```
### **Required Files:**
- `content.jpg` → The main image you want to stylize.
- `style.jpg` → The artistic image whose style will be transferred.

### **Steps:**
1. Load the content and style images.
2. Extract content and style features using **VGG19**.
3. Optimize a new image to blend both.
4. Save the final stylized image.

## 📝 Example Input & Output
### **🎨 Content & Style Images:**
```
📷 content.jpg (A landscape photo)
🖼️ style.jpg (A Van Gogh painting)
```
### **🏆 Generated Output:**
```
An image with the content of `content.jpg` but painted in the style of `style.jpg`
```

## 📂 File Structure
```
📁 neural-style-transfer/
├── neural_style_transfer.py  # Main Python script
├── README.md  # Project documentation
├── content.jpg  # Content image (to be stylized)
├── style.jpg  # Style image (artistic reference)
```
---
👨‍💻 Developed by **Naveen K**

