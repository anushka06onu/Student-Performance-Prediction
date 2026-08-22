plot_df = gpa_comparison_full['accuracy'].reset_index()
plot_df.columns = ['Model', 'Feature Set', 'Accuracy']
plot_df['Model'] = plot_df['Model']

plt.figure(figsize=(8, 5))
sns.barplot(data=plot_df, x='Model', y='Accuracy', hue='Feature Set', palette='pastel')
plt.title('Accuracy With vs. Without previous_gpa')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(DRIVE_OUTPUT_DIR, 'figures', 'gpa_feature_comparison.png'))
plt.show()