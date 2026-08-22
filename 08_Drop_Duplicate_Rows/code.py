# I am dropping any exact duplicate rows and, if there were missing values,
# I am imputing numerical columns with their median so no rows have to be
# discarded unnecessarily.
before = df.shape[0]
df = df.drop_duplicates().reset_index(drop=True)
after = df.shape[0]
print(f'Removed {before - after} duplicate rows. New shape: {df.shape}')

numeric_cols = ['attendance_percentage', 'quiz_average', 'assignment_average',
                 'midterm_score', 'participation_score', 'study_hours_per_week',
                 'previous_gpa']
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print('Missing values after cleaning:', df[numeric_cols].isnull().sum().sum())