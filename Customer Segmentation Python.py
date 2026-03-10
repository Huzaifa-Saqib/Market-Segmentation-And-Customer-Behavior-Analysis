import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_excel("shopping_trends.xlsx")

freq_map = {
    'Annually': 1,
    'Quarterly': 2,
    'Monthly': 3,
    'Fortnightly': 4,
    'Bi-Weekly': 4,
    'Weekly': 5
}

data['Frequency_num'] = data['Frequency of Purchases'].map(freq_map)
data = data.dropna(subset=['Frequency_num', 'Purchase Amount (USD)'])

X = data[['Purchase Amount (USD)', 'Frequency_num']]

kmeans = KMeans(n_clusters=3, random_state=42)
data['Value_Segment'] = kmeans.fit_predict(X)

centers = kmeans.cluster_centers_[:, 0]
labels = ['Low', 'Medium', 'High']
cluster_map = {i: labels[idx] for i, idx in enumerate(centers.argsort())}
data['Value_Segment'] = data['Value_Segment'].map(cluster_map)

data['Satisfaction'] = data['Review Rating'].apply(lambda x: 'Satisfied' if x > 3.5 else 'Unsatisfied')

print(data[['Customer ID', 'Purchase Amount (USD)', 'Frequency of Purchases', 'Value_Segment', 'Satisfaction']])

data.to_csv("Customer_Segments.csv", index=False)



counts = data['Value_Segment'].value_counts()
total = len(data)

low_value = (counts.get('Low', 0) / total) * 100
high_value = (counts.get('High', 0) / total) * 100
med_value = (counts.get('Medium', 0) / total) * 100

print("\n" + "="*30)
print(" CUSTOMER SEGMENTATION REPORT ")
print("="*30)
print(f"Total Customers Analyzed: {total}")
print(f"High-Value Customers:     {high_value:.2f}%")
print(f"Medium-Value Customers:   {med_value:.2f}%")
print(f"Low-Value Customers:      {low_value:.2f}%")
print("="*30)

# ======= Projected Revenue Increase =======

revenue_by_segment = data.groupby('Value_Segment')['Purchase Amount (USD)'].sum()

counts_by_segment = data['Value_Segment'].value_counts()

num_low = counts_by_segment.get('Low', 0)

avg_revenue = revenue_by_segment / counts_by_segment

avg_high = avg_revenue.get('High', 0)

conversion_rate = 0.5 

num_low_converted = num_low * conversion_rate

current_total_revenue = data['Purchase Amount (USD)'].sum()

revenue_lost = revenue_by_segment['Low'] * conversion_rate

revenue_gained = num_low_converted * avg_high

projected_revenue = current_total_revenue - revenue_lost + revenue_gained

percent_increase = ((projected_revenue - current_total_revenue) / current_total_revenue) * 100

print(f"\nProjected Revenue Increase: {percent_increase:.2f}%")