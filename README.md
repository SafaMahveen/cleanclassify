# cleanclassify

**cleanclassify** is a beginner-friendly Python package that helps you **clean**, **classify**, and **visualize** CSV data — all from a sleek graphical interface. Whether you're a student, data science enthusiast, or someone exploring machine learning for the first time, this tool simplifies your journey.

## What It Does

- **Cleans your dataset automatically**
  - Handles missing values
  - Drops problematic or high-cardinality columns
  - Scales numeric features
  - Encodes categorical variables

- **Runs machine learning models**
  - Trains and evaluates Logistic Regression, Random Forest, and Support Vector Classifier using `scikit-learn`

- **Visualizes model performance**
  - Shows accuracy, precision, recall, and F1-score
  - Highlights the best-performing model
  - Plots a clean bar chart using `matplotlib`

- **Requires zero coding**
  - Just load your CSV, pick the target column, and click the clean , classify buttons — that’s it!


## Installation

Install it directly from PyPI:

```bash
pip install cleanclassify
````

>  This will automatically install required dependencies:
> `pandas`, `numpy`, `scikit-learn`, and `matplotlib`.

---

##  How to Use

Launch the GUI with:

```bash
python -m cleanclassify
```
---

## 💻 Example Workflow

1. Launch the app.
2. Browse and load your CSV file.
3. Select the target column you want to predict.
4. Click **Run Cleaning** to clean and prepare your dataset.
5. Click **Run Classification** to train and evaluate models.
6. View detailed metrics and a comparison chart of model performance.

---

## What Your Data Should Look Like

* Must contain a **target column** (the label you're predicting).
* Can include both numeric and categorical features.
* Should not include long text or extremely high-cardinality columns (they’ll be automatically dropped for performance).
* If the dataset has more than 2000 rows, it will be automatically **downsampled** for memory efficiency.

---

## Under the Hood

* **`cleaner.py`** — Preprocesses data: cleans, encodes, scales, and downsamples.
* **`classify.py`** — Trains and evaluates three ML models.
* **`gui.py`** — A simple but powerful GUI built with `tkinter`.

---

## 👤 Author

Crafted with ❤️ by **Safa Mahveen**
