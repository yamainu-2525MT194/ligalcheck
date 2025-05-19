import pandas as pd

df = pd.read_csv('datasets/raw_contracts/sample_contracts.csv')
print('行数:', len(df))
print(df[['text', 'risk_level']].head(10))
print('risk_levelのユニーク値:', df['risk_level'].unique())
