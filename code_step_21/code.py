knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier(n_neighbors=7))
])
train_and_evaluate('KNN', knn_pipeline, X_train, y_train, X_test, y_test, grade_labels)

nb_pipeline = Pipeline([
    ('clf', GaussianNB())
])
train_and_evaluate('Naive Bayes', nb_pipeline, X_train, y_train, X_test, y_test, grade_labels)