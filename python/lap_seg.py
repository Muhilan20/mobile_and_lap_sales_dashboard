import pandas as pd 
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel("lap.xlsx")
print(df)

#clean_columns
df.columns = df.columns.str.strip().str.lower().str.replace(' ', "_")


print (df.columns)
cat_cols = ['product', 'brand', 'product_code', 'price',  'lap_customer_name', 
            'customer_location', 'region', 'processor_specification' ]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le 

df.to_csv("lap_cleaned.csv",index = False)

from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('lap_cleaned.csv')
print (df)
df['dispatch_date'] = pd.to_datetime(df['dispatch_date'], format= '%Y-%m-%d')

today = df['dispatch_date'].max() + pd.Timedelta(days=1)


rfm = df.groupby('lap_customer_name').agg({
    'dispatch_date': lambda x: (today - x.max()).days,  # Recency
    'product_code': 'count',                            # Frequency
    'sales': 'sum'                                      # Monetary
}).reset_index()

rfm + df.groupby('lap_customer_name').agg({ 
    "dispatch_date": lambda x : (today - x.max()).days,
    "product_code": "count",
    "sales": "sum"})
rfm.columns = ['lap_customer_name', 'Recency', 'Frequency', 'Monetary']
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(rfm_scaled)
    wcss.append(km.inertia_)
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)

# Save results
rfm.to_csv("customer_segments.csv", index=False)
print(rfm.head())



