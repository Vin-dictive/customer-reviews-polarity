# 🚀 Amazon Review Sentiment Classification using Naive Bayes

This project focuses on classifying Amazon product reviews into positive or negative sentiment using a machine learning pipeline based on the **Bernoulli Naive Bayes** algorithm. The implementation uses best practices like data subsetting, scikit-learn pipelines, and cross-validation for hyperparameter tuning.

***

## ⚙️ Project Setup and Dependencies

The project is built using Python and standard data science libraries.

### Requirements

To run the notebook, you need the following libraries:

* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib` (for potential visualizations)

### Data

The project utilizes a large Amazon review dataset, stored in the `data/` directory:

* `data/train.csv`
* `data/test.csv`

The raw files follow a structured format: `[Label, Title, Review Content]`.

***

## 💾 Data Processing and Cleaning

### Data Subset and Cleanup

Due to the size of the full training set, the initial phase focuses on a manageable subset:

1. **Column Mapping:** The raw, headerless CSV files are loaded and columns are renamed:
    * Column `0` $\rightarrow$ `label` (Sentiment: **1** or **2**)
    * Column `1` $\rightarrow$ `title`
    * Column `2` $\rightarrow$ `content` (Review Text)
2. **Feature Reduction:** The **`title`** column was dropped from both datasets.
3. **Subsetting:** The primary `train` DataFrame was limited to the **first 50,000 rows** for efficient training and cross-validation.
4. **Data Split:** The 50,000 rows were split into training and validation sets:
    * **$X_{train}, y_{train}$:** 80% (40,000 reviews) for model training.
    * **$X_{test}, y_{test}$:** 20% (10,000 reviews) for validation. *(Note: This split is from the 50k subset, not the external `data/test.csv` file)*.

### Feature Engineering Pipeline

The text data is prepared for the Naive Bayes model using a `Pipeline` combining vectorization and the classifier:

| Pipeline Step | Class | Role | Key Parameters |
| :--- | :--- | :--- | :--- |
| **Vectorization** | `CountVectorizer` | Converts text to numerical features (Bag-of-Words). | `max_features=10000`, `stop_words='english'`, **`binary=True`** |
| **Classifier** | `BernoulliNB` | The Naive Bayes model. | `alpha` (tuned below) |

The **`binary=True`** parameter in `CountVectorizer` is essential here, as it forces the features to be binary (1 if a word is present, 0 if absent), matching the assumption of the **Bernoulli Naive Bayes** classifier.

***

## 🔬 Model Training and Results

### Hyperparameter Tuning (`GridSearchCV`)

The model's smoothing parameter, **$\alpha$** (alpha), was optimized using 5-Fold Grid Search Cross-Validation (`GridSearchCV`) on the training data.

* **Model Tested:** `pipe_nb` (CountVectorizer $\rightarrow$ BernoulliNB)
* **Parameter Grid:** `{"bernoullinb__alpha": [0.1, 0.5, 1.0, 2.0, 5.0]}`
* **Scoring Metric:** `accuracy`

#### Cross-Validation Performance

The optimal $\alpha$ was found to be the default value, demonstrating good stability.

| Metric | Value |
| :--- | :--- |
| **Best $\alpha$ parameter found** | **1.0** |
| **Best Cross-Validation Accuracy** | **0.8257** |

***

## ⏭️ Future Work

To further enhance the project, the following steps are recommended:

1. **Final Test Evaluation:** Evaluate the optimized model on the completely unseen external test dataset (`data/test.csv`) to obtain an unbiased final performance score and classification report.
2. **Algorithm Comparison:** Implement and compare the performance against **Multinomial Naive Bayes** (which uses word counts/TF-IDF) to determine the best approach for this dataset.
3. **Tuning Optimization:** Utilize **RandomizedSearchCV** to explore a broader range of hyperparameters simultaneously for both the `CountVectorizer` (e.g., `ngram_range`) and the Naive Bayes $\alpha$ to achieve potentially better results faster.
