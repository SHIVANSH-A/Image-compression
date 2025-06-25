```markdown
# 📦 PCA Image Compressor (Flask App)

This Flask web app allows users to upload an image, compress it using PCA (Principal Component Analysis), visualize the explained variance, and compare original vs. compressed image size and shape.

---

## 🚀 Features

- 📤 Upload image via web UI  
- 📏 See original image size & shape  
- 🧠 Apply PCA-based compression  
- 📉 View variance explained by principal components  
- 🖼️ Downloadable compressed image  
- 📊 Visual plots:
  - Bar chart of variance per component
  - Line plot showing cumulative explained variance

---

## 📂 Folder Structure

```
pca_image_app/
├── app.py
├── utils.py
├── templates/
│   └── index.html
├── static/
│   ├── uploads/     # Original uploaded images
│   ├── outputs/     # Compressed output images
│   └── plots/       # Variance plots
```

---

## 📦 Installation

### 🔧 1. Clone or Download

```bash
git clone https://github.com/your-username/pca-image-compressor.git
cd pca-image-compressor
```

### 📦 2. Install Dependencies

```bash
pip install flask numpy pillow matplotlib scikit-learn
```

---

## ▶️ Running the App

```bash
python app.py
```

Then open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000)
