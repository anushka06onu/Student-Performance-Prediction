comparison_df = pd.DataFrame({
    name: {
        'Accuracy': res['accuracy'],
        'Precision (macro)': res['precision'],
        'Recall (macro)': res['recall'],
        'F1-score (macro)': res['f1'],
    }
    for name, res in results.items()
}).T.sort_values('Accuracy', ascending=False)

comparison_df.round(4)