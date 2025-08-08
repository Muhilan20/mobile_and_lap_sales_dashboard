import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_excel("mobile.xlsx")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print (df.columns)

# Optional: Preview columns
print("Columns:", df.columns.tolist())

# Optional: Drop unwanted columns if any
# df.drop(['unnamed:_0'], axis=1, inplace=True)  # example only

# Encode categorical features
cat_cols = ['product', 'brand', 'product_code', 'customer_name',
            'customer_location', 'region', 'processor_specification']

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le  # Save for inverse_transform if needed

# Convert date columns if not done in Excel
# df['invoice_date'] = pd.to_datetime(df['invoice_date'])



# Optional: Feature Engineering
# df['sales'] = df['quantity'] * df['price']
# df['stock_days'] = df['stock'] / df['daily_sales']  # If available

# Preview cleaned data
print(df.head())

# Save cleaned file
df.to_csv("mobile_cleaned.csv", index=False)
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("mobile_cleaned.csv")  # Your cleaned dataset

# Convert date
df['dispatch_date'] = pd.to_datetime(df['dispatch_date'], format='%Y-%m-%d')


# Set a reference date for Recency
today = df['dispatch_date'].max() + pd.Timedelta(days=1)

# RFM Calculation
rfm = df.groupby('customer_name').agg({
    'dispatch_date': lambda x: (today - x.max()).days,  # Recency
    'product_code': 'count',                            # Frequency
    'sales': 'sum'                                      # Monetary
}).reset_index()

# Rename columns
rfm.columns = ['customer_name', 'Recency', 'Frequency', 'Monetary']

# Step 2: Normalize
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Step 3: Elbow Method
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

# Choose optimal k based on elbow (say k=4)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)

# Save results
rfm.to_csv("customer_segments.csv", index=False)
print(rfm.head())







