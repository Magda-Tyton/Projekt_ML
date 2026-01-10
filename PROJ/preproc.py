from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv(
    'PROJ/autoscout24_dataset_20251108.csv',
    encoding='utf-8',
    sep=',',
    engine='python'
)

# print(df.info())

for delname in ['warranty', 'fuel_cons_highway_l100_km', 'has_warranty', 'fuel_cons_city_l100_km']:
    del df[delname]

del df['price_currency']

obj_cols = df.select_dtypes(include=["object"]).columns.tolist()

df['weight_kg'] = df['weight_kg'].str.replace(' kg', '', regex=False)
df['weight_kg'] = df['weight_kg'].str.replace(',', '', regex=False).astype('Int64')

df['mileage_km'] = df['mileage_km'].str.replace(' km', '', regex=False)
df['mileage_km'] = df['mileage_km'].str.replace(',', '', regex=False).astype('Int64')

df['ratings_average'] = df['ratings_average'].str.replace(',', '.', regex=False).astype('Float64')

obj_cols.remove('weight_kg')
obj_cols.remove('mileage_km')
obj_cols.remove('ratings_average')

num_cols = df.select_dtypes(include=["float"]).columns.tolist()
bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

# df[num_cols].hist(figsize=(16, 12), bins=30)
# plt.tight_layout()
# plt.show()

corr = df[num_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.show()
# print(df[obj_cols].nunique().sort_values(ascending=False))

# print(df.info())

# print((df.isna().mean() * 100).sort_values(ascending=False).to_string())
