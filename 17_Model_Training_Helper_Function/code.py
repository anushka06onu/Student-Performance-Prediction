results = {}

def train_and_evaluate(name, pipeline, X_tr, y_tr, X_te, y_te, labels, store_key=None):
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_te, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)

    key = store_key if store_key else name
    results[key] = {
        'model': pipeline,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'y_pred': y_pred,
    }

    print(f'--- {name} ---')
    print(f'Accuracy:  {acc:.4f}')
    print(f'Precision (macro): {prec:.4f}')
    print(f'Recall (macro):    {rec:.4f}')
    print(f'F1-score (macro):  {f1:.4f}')
    print()
    print(classification_report(y_te, y_pred, target_names=labels, zero_division=0))

    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix — {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    safe_name = name.lower().replace(' ', '_')
    plt.savefig(os.path.join(DRIVE_OUTPUT_DIR, 'figures', f'confusion_matrix_{safe_name}.png'))
    plt.show()

    return pipeline