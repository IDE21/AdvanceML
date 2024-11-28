# AdvanceML

# **Advanced Machine Learning Project**

This repository contains a comprehensive machine learning pipeline designed to classify a dataset's target variable (`status`) using advanced preprocessing, multiple classification models, and insightful visualizations. The focus is on training robust models, analyzing their performance, and visualizing both the data and results for better interpretability.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Features](#features)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Visualizations](#visualizations)
7. [Installation and Usage](#installation-and-usage)
8. [Technologies Used](#technologies-used)
9. [Contributing](#contributing)
10. [License](#license)

---

## **Project Overview**

The goal of this project is to develop machine learning models to predict the target variable (`status`) using various classification algorithms, including:
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**

Additionally, the repository includes detailed visualizations to:
- Analyze the data structure and relationships.
- Evaluate model performance.
- Understand the importance of features in classification.

---

## **Dataset Description**

### **Data Summary**
- **File**: `dataset.csv`
- **Number of Samples**: 1,195
- **Number of Features**: 24 (including the target variable `status`)
- **Target Variable**: `status` (binary classification)

### **Target Variable**
- `0`: Indicates one category (e.g., absence of a condition).
- `1`: Indicates another category (e.g., presence of a condition).

---

## **Features**

Key features in the dataset include:
- **MDVP:Fo(Hz)**: Average vocal fundamental frequency.
- **MDVP:Fhi(Hz)**: Maximum vocal fundamental frequency.
- **MDVP:Flo(Hz)**: Minimum vocal fundamental frequency.
- **Jitter(%)**: Measure of frequency variation.
- **Shimmer**: Measure of amplitude variation.

The dataset underwent extensive preprocessing to handle missing values, scale features, and prepare data for machine learning models.

---

## **Methodology**

### **Steps**
1. **Data Preprocessing**:
   - Handling missing values via mean imputation.
   - Standardizing numerical features using `StandardScaler`.
2. **Model Training**:
   - Decision Tree Classifier.
   - Random Forest Classifier (Ensemble).
   - Gradient Boosting Classifier (Ensemble).
3. **Evaluation**:
   - Accuracy, precision, recall, F1-score.
   - Confusion matrix and ROC curves.
4. **Visualizations**:
   - Data distributions and correlations.
   - Feature importance analysis.
   - Comparison of model performance.

### **Key Techniques**
- **Ensemble Learning**: Leveraged Random Forest and Gradient Boosting for better generalization.
- **Visualization**: Detailed plots for insights into data and models.

---

## **Results**

### **Model Performance**
| Metric              | Decision Tree | Random Forest | Gradient Boosting |
|---------------------|---------------|---------------|-------------------|
| **Training Accuracy** | 100%          | 100%          | 100%              |
| **Testing Accuracy**  | 97.49%        | 99.16%        | 98.33%            |
| **Precision (Class 1)**| 86%           | 93%           | 92%               |
| **Recall (Class 1)**   | 92%           | 100%          | 92%               |
| **F1-Score (Class 1)** | 89%           | 96%           | 92%               |

### **Key Insights**
- Random Forest achieved the best overall performance, especially in correctly identifying the minority class.
- Gradient Boosting followed closely, offering competitive accuracy and precision.
- Decision Tree was accurate but lacked the robustness of ensemble methods.

---

## **Visualizations**

This repository includes the following visualizations:
1. **Data Analysis**:
   - Missing Values Heatmap
   - Correlation Heatmap
   - Target Variable Distribution
2. **Feature Insights**:
   - Pairplots of feature relationships.
   - KDE plots of individual feature distributions.
   - Feature Importance bar charts.
3. **Model Performance**:
   - Confusion Matrix heatmaps for all models.
   - ROC Curves for ensemble models.
   - Learning Curves for model training performance.

---

## **Installation and Usage**

### **Prerequisites**
- Python 3.8 or later.
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`.

### **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/advanced-ml-project.git
   cd advanced-ml-project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook
   ```
4. Load the `AdvanceML.ipynb` notebook and execute the cells.

---

## **Technologies Used**

- **Programming Language**: Python
- **Machine Learning Framework**: Scikit-learn
- **Data Visualization**: Matplotlib, Seaborn
- **Notebook Environment**: Jupyter Notebook

---

## **Contributing**

Contributions are welcome! If you find a bug or have suggestions for improvement:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.
