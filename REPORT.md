# Machine Learning Project Report

## Tasks 1 and 3

**Student Name:** _Shreya Chel _________________________  
**Roll Number:** __BUR/MCS/2024/018 _ _______________________  
**Department:** ____COMPUTER SCIENCE ________________________  
**Course:** __M.SC IN COMPUTER SCIENCE ______________________________  
**Academic Year:** __2024-26 _______________________  

---

# Abstract

This project implements two machine-learning applications. The first application is a credit scoring model that classifies customers according to creditworthiness using financial and payment-history features. Logistic Regression, Decision Tree, and Random Forest classifiers are trained and compared using accuracy, precision, recall, F1-score, and ROC-AUC.

The second application is handwritten digit recognition using a Convolutional Neural Network (CNN). The model is trained on the MNIST dataset, which contains grayscale images of handwritten digits from 0 to 9. Model performance is evaluated using test accuracy, loss, a classification report, and a confusion matrix.

Together, the two tasks demonstrate machine learning on both structured/tabular data and image data.

---

# 1. Introduction

Machine learning allows computer systems to learn useful patterns from data and make predictions without being explicitly programmed for every individual case.

This project focuses on two different supervised-learning problems:

1. Credit scoring, a binary classification problem using structured financial data.
2. Handwritten digit recognition, a multi-class image classification problem using deep learning.

The projects demonstrate the complete machine-learning lifecycle: data preparation, feature processing, model development, evaluation, visualization, and model saving.

---

# 2. Objectives

## Task 1: Credit Scoring

- Predict creditworthiness from financial information.
- Perform feature engineering.
- Compare multiple classification algorithms.
- Measure performance with classification metrics.
- Analyze feature importance.
- Save the best-performing model.

## Task 3: Handwritten Character Recognition

- Recognize handwritten digits.
- Preprocess grayscale images.
- Develop a CNN.
- Train and validate the network.
- Evaluate performance on the MNIST test set.
- Visualize predictions and errors.
- Save the trained model.

---

# 3. Task 1: Credit Scoring Model

## 3.1 Problem Statement

Financial institutions need methods for estimating the risk associated with lending. A machine-learning classifier can learn relationships between financial characteristics and a historical creditworthiness label.

The objective of this project is to predict:

- `1`: creditworthy
- `0`: not creditworthy

The model is intended as an educational demonstration. A real lending system would require validated financial data, fairness testing, regulatory review, security controls, explainability, and human oversight.

## 3.2 Dataset

The included script automatically generates a reproducible synthetic dataset if `credit_data.csv` is not present. The demonstration dataset contains 5,000 records and the following variables:

| Feature | Description |
|---|---|
| age | Customer age |
| annual_income | Annual income |
| employment_years | Years of employment |
| debt | Existing debt |
| number_of_open_accounts | Number of open credit accounts |
| payment_history_score | Payment-history score |
| late_payments | Number of late payments |
| credit_utilization | Credit utilization ratio |
| loan_amount | Requested loan amount |
| previous_defaults | Previous default count |
| savings | Savings |
| creditworthy | Target variable |

**Academic note:** Synthetic data is suitable for demonstrating the pipeline but should not be presented as real-world evidence. If your instructor requires a public dataset, use the approved dataset and place its CSV in the Task 1 folder.

## 3.3 Data Preprocessing

The pipeline performs:

- Missing-value handling using median imputation.
- Standardization for numerical features.
- Stratified train/test splitting.

The test set is kept separate from model training.

## 3.4 Feature Engineering

Three additional variables are created:

- Debt-to-income ratio
- Loan-to-income ratio
- Savings-to-income ratio

These ratios can provide more meaningful information than raw monetary values alone.

## 3.5 Algorithms

### Logistic Regression

Logistic Regression estimates the probability of belonging to the positive class. It is useful as a simple and interpretable baseline.

### Decision Tree

A Decision Tree learns a sequence of feature-based rules. It can capture non-linear relationships and is easy to visualize conceptually.

### Random Forest

Random Forest combines many decision trees. The ensemble approach can improve generalization and provide feature-importance estimates.

## 3.6 Evaluation Metrics

### Accuracy

The proportion of all predictions that are correct.

### Precision

Of the cases predicted as positive, precision measures how many are actually positive.

### Recall

Recall measures how many actual positive cases were correctly identified.

### F1-score

F1-score is the harmonic mean of precision and recall.

### ROC-AUC

ROC-AUC measures how well the model ranks positive cases above negative cases across classification thresholds.

---

# 4. Task 1 Results

Run:

```bash
python task1_credit_scoring/credit_scoring.py
```

The program creates `model_comparison.csv` and several charts.

Paste the actual output below:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.550 | 0.5447 | 0.5534 | 0.5491 | 0.5671 |
| Decision Tree | 0.543 | 0.5358 | 0.5737 | 0.5541 | 0.5618 |
| Random Forest | 0.561| 0.5581 | 0.5434 | 0.5507 | 0.5726 |

### Best model

**Best model according to ROC-AUC:** ___Random Forest (ROC-AUC: 0.5726) _______________

### Discussion

The best model should be selected using the metric most appropriate for the project objective. For a credit-risk application, accuracy alone is not sufficient because different types of errors can have different consequences. Precision, recall, F1-score, and ROC-AUC should therefore be considered together.

The Random Forest feature-importance output can also be used to discuss which variables contributed most strongly to its predictions.

---

# 5. Task 3: Handwritten Digit Recognition

## 5.1 Problem Statement

Handwritten digit recognition is a standard image-classification problem. The goal is to assign each 28x28 grayscale image to one of ten classes:

`0, 1, 2, 3, 4, 5, 6, 7, 8, 9`

## 5.2 Dataset: MNIST

MNIST contains:

- 60,000 training images
- 10,000 test images
- 10 digit classes
- 28x28 grayscale images

Each image represents a handwritten digit.

## 5.3 Preprocessing

Pixel values originally range from 0 to 255. They are normalized to the range 0 to 1.

The image shape is changed from:

```text
28 x 28
```

to:

```text
28 x 28 x 1
```

The final dimension represents the grayscale channel.

## 5.4 CNN Architecture

The network contains:

1. Convolutional layer with 32 filters
2. Max-pooling layer
3. Convolutional layer with 64 filters
4. Max-pooling layer
5. Flatten layer
6. Dense layer with 128 neurons
7. Dropout layer
8. Output layer with 10 neurons and softmax activation

The CNN automatically learns spatial patterns such as edges, curves, and combinations of shapes.

## 5.5 Training

The model uses:

- Optimizer: Adam
- Loss: Sparse categorical cross-entropy
- Metric: Accuracy
- Batch size: 128
- Maximum epochs: 10
- Validation split: 10%
- Early stopping based on validation loss

---

# 6. Task 3 Results

Run:

```bash
python task3_handwritten_recognition/handwritten_cnn.py
```

Paste the generated test result here:

**Test loss:** ___0.0310 _______________

**Test accuracy:** ____98.94% ______________

The script also produces:

- Accuracy curve
- Loss curve
- Confusion matrix
- Sample predictions
- Classification report
- Saved CNN model

## Classification Report

Paste the contents of:

```text
task3_handwritten_recognition/outputs/classification_report.txt
```

here.

## Discussion

A successful CNN should classify most MNIST images correctly. The confusion matrix helps identify which digit pairs are most frequently confused. Digits with similar handwritten shapes can be harder to distinguish.

The training and validation curves can be used to determine whether the model is learning appropriately or beginning to overfit.

---

# 7. Comparison of the Two Tasks

| Property | Credit Scoring | Handwritten Recognition |
|---|---|---|
| Data type | Tabular | Image |
| Learning type | Supervised | Supervised |
| Main problem | Binary classification | Multi-class classification |
| Main models | Logistic Regression, Decision Tree, Random Forest | CNN |
| Main dataset | Financial/credit records | MNIST |
| Important preprocessing | Imputation, scaling, feature engineering | Normalization, reshaping |
| Main metrics | Precision, Recall, F1, ROC-AUC | Accuracy, precision, recall, F1 |
| Main visualization | ROC curve, confusion matrix | Learning curves, confusion matrix |

---

# 8. Limitations

## Credit Scoring

- The included fallback dataset is synthetic.
- Synthetic labels do not represent real lending decisions.
- A real deployment would require much more extensive validation.
- Credit decisions can create fairness and regulatory concerns.
- Model predictions should not be treated as automatic financial decisions.

## Handwritten Recognition

- MNIST is relatively simple compared with real handwriting.
- Real-world images may contain different backgrounds, sizes, rotations, and writing styles.
- Performance on MNIST does not guarantee equivalent performance in another application.

---

# 9. Future Enhancements

## Credit Scoring

- Use a validated public or institutional dataset.
- Apply cross-validation and hyperparameter tuning.
- Compare additional models such as Gradient Boosting.
- Add calibration and probability-threshold analysis.
- Perform fairness and subgroup evaluation.
- Add explainability techniques.

## Handwritten Recognition

- Extend from digits to EMNIST characters.
- Add data augmentation.
- Experiment with deeper CNN architectures.
- Recognize complete words using sequence models.
- Build a small user interface for uploading a handwritten image.

---

# 10. Conclusion

This project demonstrates two different applications of supervised machine learning. The credit scoring task shows how structured financial information can be processed and classified using traditional machine-learning algorithms. The handwritten digit task demonstrates how CNNs can learn visual features directly from image pixels.

The combination provides practical experience with preprocessing, feature engineering, model training, evaluation, visualization, and model persistence. The project also highlights the importance of selecting evaluation metrics according to the real purpose of a machine-learning system.

---

# 11. References

1. LeCun, Y., Cortes, C., & Burges, C. J. C. MNIST handwritten digit database.
2. Pedregosa et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research.
3. Chollet, F. et al. Keras documentation.
4. Géron, A. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.
