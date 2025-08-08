import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('lap_cleaned.csv')  # Replace with your actual filename

frs = ['price', 'ram', 'rom', 'ssd', 'quantity_sold', 'sales']
X = df[frs].fillna(0)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Now df contains the cluster info!

print(df)

df.to_csv('lap_seg.csv', index=False)
