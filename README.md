<div align="center">

# 🌸 Data Classification Using AI

### A basic supervised learning project — from raw data to a trained, evaluated classifier

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Live Demo](web/index.html) · [View Notebook](notebook/Data_Classification.ipynb) · [Report Issue](https://github.com/AnasHasnain333/data-classification-ai/issues) · [Author](#-author)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Requirements Covered](#-requirements-covered)
- [Dataset](#-dataset)
- [Model](#-model)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Pipeline Explained](#-pipeline-explained)
- [Possible Extensions](#-possible-extensions)
- [Author](#-author)
- [License](#-license)

---

## 🎯 Overview

**Project 2** in a hands-on machine learning practice series. The goal is to
build a complete, minimal, end-to-end **classification pipeline**: load a
dataset, split it correctly, train a simple algorithm, and evaluate it
honestly — the foundation every larger ML project is built on.

## 🌐 Live Demo

Open [`web/index.html`](web/index.html) directly in any browser — no server,
no build step. It's a self-contained page that presents the whole project
visually: the pipeline, the evaluation results, and a **live interactive
classifier** where you can drag sliders to describe a flower's measurements
and see the predicted species computed in real time, right in the browser
(a small JavaScript K-Nearest Neighbors implementation, using the same
150 training specimens as the Python model).

```bash
# from the project root
open web/index.html        # macOS
start web/index.html       # Windows
xdg-open web/index.html    # Linux
```

## ✅ Requirements Covered

| Requirement | Status |
|---|---|
| Load and understand a dataset | ✅ |
| Split data into training and testing sets | ✅ |
| Apply a simple classification algorithm | ✅ |
| Evaluate model performance | ✅ |

**Key skills demonstrated:** data handling (`pandas`), supervised learning
basics, model training & evaluation (`scikit-learn`).

---

## 📊 Dataset

This project uses the classic **[Iris flower dataset](https://archive.ics.uci.edu/dataset/53/iris)**:

| Property | Value |
|---|---|
| Samples | 150 |
| Features | 4 numeric — sepal length, sepal width, petal length, petal width (cm) |
| Classes | 3 — `setosa`, `versicolor`, `virginica` |
| Missing values | None |

The dataset ships with the repo at [`data/iris.csv`](data/iris.csv), so the
project runs end-to-end with **no external downloads**.

## 🤖 Model

A **K-Nearest Neighbors (KNN)** classifier (`n_neighbors=5`) — a simple,
intuitive, distance-based algorithm well suited to a small, clean,
well-separated dataset like Iris.

Features are standardized with `StandardScaler` before training so every
feature contributes equally to the distance calculation.

## 📈 Results

| Metric | Score |
|---|---|
| **Accuracy** | ~93% |
| Train / test split | 80% / 20% (stratified) |

<table>
<tr>
<td align="center"><b>Confusion Matrix</b><br><img src="images/confusion_matrix.png" width="380"></td>
<td align="center"><b>Feature Relationships</b><br><img src="images/pairplot.png" width="420"></td>
</tr>
</table>

> Exact numbers may vary slightly between runs, since accuracy depends on
> the random train/test split.

---

## 📁 Project Structure

```
data-classification-ai/
├── data/
│   └── iris.csv                    # Dataset used for training/testing
├── notebook/
│   └── Data_Classification.ipynb   # Step-by-step notebook walkthrough
├── src/
│   └── classify.py                 # Full pipeline as a runnable script
├── models/
│   └── knn_model.joblib            # Saved trained model (generated on run)
├── images/
│   ├── confusion_matrix.png        # Generated evaluation chart
│   └── pairplot.png                # Generated exploratory chart
├── web/
│   ├── index.html                  # Interactive browser demo (live classifier)
│   └── images/                     # Copies of charts used by the web page
├── requirements.txt                # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AnasHasnain333/data-classification-ai.git
cd data-classification-ai
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the script

```bash
python src/classify.py
```

This will:
- Load and print a summary of the dataset
- Generate an exploratory pairplot (`images/pairplot.png`)
- Split the data 80/20 into train/test sets
- Train a KNN classifier
- Print accuracy and a classification report
- Save a confusion matrix plot (`images/confusion_matrix.png`)
- Save the trained model (`models/knn_model.joblib`)

### 5. Or explore the notebook

```bash
jupyter notebook notebook/Data_Classification.ipynb
```

### 6. Or just open the live demo

```bash
open web/index.html
```

---

## 🔍 Pipeline Explained

1. **Load & understand the data** — read `iris.csv` into a pandas
   DataFrame; inspect shape, column types, class balance, and summary
   statistics.
2. **Split the data** — separate features (`X`) from the target
   (`species`), encode text labels into integers, and split into 80%
   training / 20% testing with `train_test_split` (stratified so each
   class is proportionally represented in both sets).
3. **Preprocess** — scale features with `StandardScaler` so no single
   feature dominates the distance calculation used by KNN.
4. **Train** — fit a `KNeighborsClassifier` on the training set.
5. **Evaluate** — predict on the held-out test set and report accuracy,
   precision/recall/F1 per class, and a confusion matrix.

## 💡 Possible Extensions

- Swap KNN for Logistic Regression, Decision Tree, or SVM and compare results
- Try a different small dataset (e.g. Wine, Breast Cancer, Titanic)
- Add cross-validation instead of a single train/test split
- Tune `n_neighbors` with a grid search
- Wrap the trained model in a simple Flask/FastAPI endpoint for predictions
- Deploy `web/index.html` with GitHub Pages for a shareable public link

---

## 👤 Author

**Abdul Rafay**
Artificial intelligence intern

[![GitHub](https://github.com/Abdulrafay662/)
[![LinkedIn](https://www.linkedin.com/in/abdul-rafay-276408257)

## 📄 License

This project is licensed under the [MIT License](LICENSE) — free to use,
modify, and distribute with attribution.

---

<div align="center">

⭐ If you found this project helpful, consider giving it a star!

</div>
