# ============================================================
# DOWNLOAD DATASET → SAVE TO DRIVE → RENAME COLUMNS → LOAD DF
# ============================================================

from google.colab import drive
drive.mount("/content/drive")

import os
import pandas as pd

# Raw CSV file from the original GitHub repository
dataset_url = (
    "https://raw.githubusercontent.com/"
    "ovxncdev/student-analytics-platform/"
    "main/data-processing/student_data.csv"
)

# Google Drive folder and file location
drive_folder = (
    "/content/drive/MyDrive/"
    "student_performance_prediction"
)

csv_path = os.path.join(
    drive_folder,
    "student_performance_dataset.csv"
)

# Create the Drive folder if it does not exist
os.makedirs(drive_folder, exist_ok=True)

print("Downloading dataset from GitHub...")

# Download the dataset
df = pd.read_csv(dataset_url)

print("Original dataset downloaded.")
print("Original shape:", df.shape)
print("Original columns:", df.columns.tolist())

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

# Rename target columns to match the notebook
df = df.rename(columns={
    "final_score": "score",
    "final_grade": "grade"
})

# Required columns for the notebook
required_columns = [
    "attendance_percentage",
    "quiz_average",
    "assignment_average",
    "midterm_score",
    "participation_score",
    "study_hours_per_week",
    "previous_gpa",
    "score",
    "grade",
    "pass_fail"
]

# Check for missing columns
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"The dataset is missing these columns: {missing_columns}"
    )

# Save the corrected dataset to Google Drive
df.to_csv(csv_path, index=False)

print("\nDataset downloaded and prepared successfully.")
print("Saved to:", csv_path)
print("Dataset shape:", df.shape)
print("Final columns:", df.columns.tolist())

print("\nGrade distribution:")
print(df["grade"].value_counts().sort_index())

display(df.head())