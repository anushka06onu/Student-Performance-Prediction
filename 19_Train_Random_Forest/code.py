rf_pipeline = Pipeline([
    ('clf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
])
train_and_evaluate('Random Forest', rf_pipeline, X_train, y_train, X_test, y_test, grade_labels)