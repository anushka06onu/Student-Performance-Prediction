logreg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])
train_and_evaluate('Logistic Regression', logreg_pipeline, X_train, y_train, X_test, y_test, grade_labels)