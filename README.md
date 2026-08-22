# Student Performance Prediction

This project focuses on analyzing student data and predicting student performance grades based on various demographic, academic, and behavioral features. The dataset was cleaned, explored through data visualization, and subsequently used to train and compare several machine learning classification models.

## Architecture

The project follows a standard machine learning pipeline architecture, structured into sequential modular steps:

1. **Data Ingestion & Setup:** Linking to Google Drive and downloading the dataset (`01` - `03`).
2. **Data Inspection & Cleaning:** Checking data shapes, types, missing values, and handling duplicates (`04` - `08`).
3. **Exploratory Data Analysis (EDA):** Statistical summaries, distribution analysis, and generating visual insights to understand feature relationships with the target variable (`09` - `16`).
4. **Model Development & Training:** Building scikit-learn pipelines with standard scalers and training 5 different classifiers (`17` - `21`).
5. **Evaluation & Comparison:** Aggregating metrics (Accuracy, Precision, Recall, F1-score) into a comparative dataframe and visualizing the best models (`22` - `24`).
6. **Feature Selection Analysis:** Testing the isolated effect of `previous_gpa` on model performance (`25` - `26`).
7. **Serialization:** Saving the best-performing model pipeline and label encoder using `joblib` for future inference (`27`).

## Methodology

### 1. Data Preprocessing & Cleaning
- **Handling Missing Values:** Missing values were imputed appropriately depending on the data type (e.g., mean imputation for numerical, mode for categorical).
- **Removing Duplicates:** Exact duplicate rows were identified and dropped to prevent data leakage and bias.
- **Feature Encoding & Scaling:** Categorical variables were encoded into numerical formats, and numerical features (e.g., `study_hours_per_week`, `attendance_percentage`) were standardized using `StandardScaler` to ensure distance-based models (like SVM and KNN) perform optimally.

### 2. Exploratory Data Analysis (EDA)
- **Target Distribution:** We first investigated the distribution of grades to ensure the classes were not heavily imbalanced.
- **Bivariate Analysis:** Boxplots were generated to analyze the relationships between `study_hours_per_week`, `attendance_percentage`, `midterm_score`, `previous_gpa` and the final `grade`.
- **Multicollinearity Check:** A correlation heatmap was generated to understand relationships between numerical features, ensuring no two features were highly collinear.

### 3. Machine Learning Modeling
- **Pipelines:** We developed pipelines incorporating `StandardScaler` and the respective classification models to prevent data leakage during cross-validation/testing.
- **Models Trained:** 
  - **Logistic Regression:** For a strong linear baseline.
  - **Random Forest:** To capture non-linear interactions.
  - **Support Vector Machine (SVM):** For complex decision boundaries.
  - **K-Nearest Neighbors (KNN):** As a distance-based baseline.
  - **Naive Bayes:** To leverage conditional probability.
- **Evaluation:** Models were evaluated on the test set using Accuracy, Precision (macro), Recall (macro), and F1-score (macro).

### 4. Feature Selection Analysis
- We isolated the `previous_gpa` feature to assess its predictive power. We trained the best models with and without this feature to quantify how much historical performance data contributes to predicting the current course grade.

## Technologies Used

- **Python**: Core programming language.
- **Pandas & NumPy**: Data manipulation and numerical operations.
- **Matplotlib & Seaborn**: Data visualization.
- **Scikit-Learn**: Machine learning modeling, preprocessing (`StandardScaler`), evaluation metrics, and pipeline creation.
- **Joblib**: Model serialization.

## Results Table

The models were evaluated based on their performance across various metrics. Here is the summary of their performance:

| Model | Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) |
|-------|----------|-------------------|----------------|------------------|
| **Naive Bayes** | 0.82 | ~0.82 | ~0.82 | ~0.82 |
| **Logistic Regression** | 0.81 | ~0.81 | ~0.81 | ~0.81 |
| **SVM** | 0.80 | ~0.80 | ~0.80 | ~0.80 |
| **Random Forest** | 0.79 | ~0.79 | ~0.79 | ~0.79 |
| **KNN** | 0.72 | ~0.72 | ~0.72 | ~0.72 |

*Naive Bayes achieved the highest accuracy (82%), followed closely by Logistic Regression.*

## Key Figures and Visualizations

Below are key visualizations generated during the Exploratory Data Analysis and Model Evaluation phases. All images are stored in the `results_images/` directory.

### Exploratory Data Analysis

<div align="center">
  <h4>Distribution of Grades</h4>
  <img src="results_images/Distribution_of_Grades.png" width="600"/>
  <br><br>

  <h4>Study Hours per Week vs. Grade</h4>
  <img src="results_images/Study_Hours_vs_Grade.png" width="600"/>
  <br><br>

  <h4>Attendance Percentage vs. Grade</h4>
  <img src="results_images/Attendance_Percentage_vs_Grade.png" width="600"/>
  <br><br>

  <h4>Midterm Score vs. Grade</h4>
  <img src="results_images/Midterm_Score_vs_Grade.png" width="600"/>
  <br><br>

  <h4>Previous GPA vs. Grade</h4>
  <img src="results_images/Previous_GPA_vs_Grade.png" width="600"/>
  <br><br>

  <h4>Correlation Heatmap</h4>
  <img src="results_images/Correlation_Heatmap.png" width="600"/>
</div>

<br>

### Model Evaluation (Confusion Matrices)

<div align="center">
  <h4>Logistic Regression</h4>
  <img src="results_images/Logistic_Regression_Confusion_Matrix.png" width="500"/>
  <br><br>

  <h4>Random Forest</h4>
  <img src="results_images/Random_Forest_Confusion_Matrix.png" width="500"/>
  <br><br>

  <h4>SVM</h4>
  <img src="results_images/SVM_Confusion_Matrix.png" width="500"/>
  <br><br>

  <h4>KNN</h4>
  <img src="results_images/KNN_Confusion_Matrix.png" width="500"/>
  <br><br>

  <h4>Naive Bayes</h4>
  <img src="results_images/Naive_Bayes_Confusion_Matrix.png" width="500"/>
</div>

<br>

### Final Comparisons

<div align="center">
  <h4>Model Comparison Across Metrics</h4>
  <img src="results_images/Model_Comparison_Across_Metrics.png" width="700"/>
  <br><br>

  <h4>Accuracy Comparison: With vs. Without previous_gpa</h4>
  <img src="results_images/Feature_Selection_GPA_Comparison.png" width="700"/>
</div>

## Repository Structure

The code from the original Jupyter Notebook has been modularized into sequential steps:
- `01_Google_Drive_Setup.py` to `28_Final_Empty_Cell.py`: These 28 independent Python scripts represent the sequential steps of the analysis pipeline.
- `results_images/`: Contains all the figures, plots, and confusion matrices generated during the analysis.

## Conclusion

The analysis confirmed that midterm scores, quiz averages, and assignment averages have the strongest correlation with the final grade. Long-term indicators like `previous_gpa` also provided a small boost in model accuracy. The `Naive Bayes` and `Logistic Regression` models proved to be the most effective at classifying student performance in this dataset.
