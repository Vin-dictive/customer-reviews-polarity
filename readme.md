# 🚀 Amazon Review Sentiment Classification (Bernoulli Naive Bayes)

This project implements a machine learning pipeline to classify Amazon product reviews into positive (Label 2) or negative (Label 1) sentiment. The core achievement is using an **expanded Grid Search** to simultaneously tune the **vectorization method** and the **Bernoulli Naive Bayes** classifier.

---

## ⚙️ Project Configuration and Data

### Data Details

* **Training Data Subset:** The main training set was limited to the **first 100,000 rows** (`CHUNK_SIZE = 100000`).
* **Target Balance:** The 100k subset shows near-perfect balance, which is ideal for classification:
  * Label **2** (Positive): 51,267 reviews
  * Label **1** (Negative): 48,733 reviews
* **Evaluation Data:** The final model was tested on the **full, external test dataset** (`data/test.csv`), which contains **400,000** reviews.

### Feature Engineering Pipeline

The final, best model was found using a pipeline structure that compared two distinct vectorization methods (Count and TF-IDF), optimized for the Bernoulli Naive Bayes classifier.

| Pipeline Step | Class | Role | Key Configuration |
| :--- | :--- | :--- | :--- |
| **Vectorizer** | `TfidfVectorizer` (Best) | Converts text to numerical features. | **`binary=True`** (Required for BernoulliNB) |
| **Classifier** | `BernoulliNB` | Binary Naive Bayes model. | $\mathbf{\alpha}$ (Smoothing parameter) |

---

## 🔬 Optimized Model Performance

### 1. Cross-Validation & Hyperparameter Tuning

The **Expanded Grid Search** was performed on the 100,000-row training subset (5-Fold Cross-Validation) to find the best combination of vectorizer type, feature size, N-gram range, and $\alpha$.

#### **🏆 Best Model Configuration (from Grid Search)**

| Parameter | Value |
| :--- | :--- |
| **Vectorization Method** | `TfidfVectorizer` |
| **Feature Limit (`max_features`)** | **20,000** |
| **N-gram Range** | **(1, 2)** (Unigrams and Bigrams) |
| **Smoothing ($\alpha$)** | **0.1** |
| **Mean CV Accuracy** | **0.8399** |
| **Mean Train Accuracy** | **0.8634** |

***Interpretation:*** The best model uses **TF-IDF** with both single words and two-word phrases up to 20,000 features, slightly outperforming the pure CountVectorizer approach. The **Train Score (0.8634)** is very close to the **CV Score (0.8399)**, indicating the model generalizes well with minimal overfitting.

---

### 2. Final Evaluation on External Test Set

The best model from the Grid Search was applied to the entire, unseen $\mathbf{400,000}$ reviews in `data/test.csv`.

#### **✅ Final Test Results**

| Metric | Score |
| :--- | :--- |
| **Total Test Reviews** | 400,000 |
| **Final Test Accuracy** | **0.8387** |
| **F1-Score (Macro Avg)** | **0.84** |

#### **Classification Report Summary**

| Label | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **1 (Negative)** | 0.85 | 0.83 | 0.84 | 200,000 |
| **2 (Positive)** | 0.83 | 0.85 | 0.84 | 200,000 |

***Conclusion:*** The final model achieves a strong and balanced accuracy of **~83.9%** on the massive test set, demonstrating the effectiveness of the tuned Bernoulli Naive Bayes pipeline for this sentiment classification task.

---

## ⏭️ Next Steps

1. **Multinomial Naive Bayes Comparison:** Since TF-IDF with `binary=True` was the winner here, the next experiment should be running the same Grid Search with **`MultinomialNB`** and switching the vectorizers back to their standard non-binary setting (allowing word counts and TF-IDF scores). This is the conventional approach for text and could potentially yield a small performance gain.
2. **Model Persistence:** Save the `search.best_estimator_` object using `joblib` or `pickle` so the model can be deployed without retraining.
