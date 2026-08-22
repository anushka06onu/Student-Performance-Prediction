best_acc_model = comparison_df['Accuracy'].idxmax()
best_f1_model = comparison_df['F1-score (macro)'].idxmax()
print(f'Highest accuracy: {best_acc_model} ({comparison_df.loc[best_acc_model, "Accuracy"]:.4f})')
print(f'Highest F1-score: {best_f1_model} ({comparison_df.loc[best_f1_model, "F1-score (macro)"]:.4f})')

# Per-class F1-score for the best model, to see which grade is hardest to predict.
best_pred = results[best_f1_model]['y_pred']
report_dict = classification_report(y_test, best_pred, target_names=grade_labels,
                                     output_dict=True, zero_division=0)
per_class_f1 = {g: report_dict[g]['f1-score'] for g in grade_labels}
hardest_grade = min(per_class_f1, key=per_class_f1.get)
print()
print('Per-class F1-score (best model):', {k: round(v, 3) for k, v in per_class_f1.items()})
print(f'Grade that was hardest to predict: {hardest_grade}')