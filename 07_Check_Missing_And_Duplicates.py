print('Missing values per column:')
print(df.isnull().sum())
print()
print('Total duplicate rows:', df.duplicated().sum())