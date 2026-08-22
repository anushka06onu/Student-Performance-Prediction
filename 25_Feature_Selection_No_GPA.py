feature_cols_no_gpa = [c for c in feature_cols if c != 'previous_gpa']
X_no_gpa = df[feature_cols_no_gpa].copy()

X_train_ng, X_test_ng, y_train_ng, y_test_ng = train_test_split(
    X_no_gpa, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

results_no_gpa = {}

def evaluate_quietly(name, pipeline, X_tr, y_tr, X_te, y_te, store):
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    store[name] = {
        'accuracy': accuracy_score(y_te, y_pred),
        'precision': precision_score(y_te, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_te, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_te, y_pred, average='macro', zero_division=0),
    }

core_models = {
    'Logistic Regression': Pipeline([('scaler', StandardScaler()),
                                      ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
    'Random Forest': Pipeline([('clf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))]),
    'SVM': Pipeline([('scaler', StandardScaler()),
                      ('clf', SVC(kernel='rbf', random_state=RANDOM_STATE))]),
}

# With previous_gpa (reusing the metrics already computed above)
with_gpa = {name: {'accuracy': results[name]['accuracy'],
                    'precision': results[name]['precision'],
                    'recall': results[name]['recall'],
                    'f1': results[name]['f1']}
            for name in core_models}

# Without previous_gpa
for name, pipeline in core_models.items():
    evaluate_quietly(name, pipeline, X_train_ng, y_train_ng, X_test_ng, y_test_ng, results_no_gpa)

gpa_comparison = pd.DataFrame({
    (name, 'With previous_gpa'): with_gpa[name] for name in core_models
})
gpa_comparison_no = pd.DataFrame({
    (name, 'Without previous_gpa'): results_no_gpa[name] for name in core_models
})
gpa_comparison_full = pd.concat([gpa_comparison, gpa_comparison_no], axis=1).T
gpa_comparison_full.round(4)