best_model_pipeline = results[best_f1_model]['model']

joblib.dump(best_model_pipeline, os.path.join(DRIVE_OUTPUT_DIR, 'models', 'best_grade_classifier.pkl'))
joblib.dump(label_encoder, os.path.join(DRIVE_OUTPUT_DIR, 'models', 'label_encoder.pkl'))
comparison_df.to_csv(os.path.join(DRIVE_OUTPUT_DIR, 'model_comparison.csv'))

print('Saved best model to:', os.path.join(DRIVE_OUTPUT_DIR, 'models', 'best_grade_classifier.pkl'))
print('Saved model comparison table to:', os.path.join(DRIVE_OUTPUT_DIR, 'model_comparison.csv'))