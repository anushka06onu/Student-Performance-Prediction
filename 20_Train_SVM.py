svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', random_state=RANDOM_STATE))
])
train_and_evaluate('SVM', svm_pipeline, X_train, y_train, X_test, y_test, grade_labels)