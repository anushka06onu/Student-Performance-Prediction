feature_cols = ['attendance_percentage', 'quiz_average', 'assignment_average',
                 'midterm_score', 'participation_score', 'study_hours_per_week',
                 'previous_gpa']

X = df[feature_cols].copy()
y_raw = df['grade'].copy()

# Encoding the target labels (A, B, C, D) into integers for the classifiers.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
grade_labels = label_encoder.classes_
print('Grade classes:', list(grade_labels))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print('Training set shape:', X_train.shape)
print('Test set shape:', X_test.shape)