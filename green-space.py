import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from google.colab import files
uploaded = files.upload()

data = pd.read_csv('city_data.csv')
print(data.head())

# Normalize data for better clustering
data['population_density'] = (data['population_density'] - data['population_density'].mean()) / data['population_density'].std()
data['land_value'] = (data['land_value'] - data['land_value'].mean()) / data['land_value'].std()
data['existing_green_space'] = (data['existing_green_space'] - data['existing_green_space'].mean()) / data['existing_green_space'].std()

# Create new features
data['green_space_ratio'] = data['existing_green_space'] / (data['land_value'] + 1e-5)
data['high_density'] = (data['population_density'] > data['population_density'].mean()).astype(int)

# Encode categorical variable using one-hot encoding
data = pd.get_dummies(data, columns=['land_use'])

# Clustering using KMeans
features = data[['population_density', 'land_value', 'existing_green_space',
                 'green_space_ratio', 'high_density'] +
                [col for col in data.columns if 'land_use_' in col]]

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled_features)
data['pca_x'] = X_pca[:, 0]
data['pca_y'] = X_pca[:, 1]

best_n_clusters = 0
best_silhouette = -1

for n_clusters in range(3, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_pca)
    silhouette = silhouette_score(X_pca, labels)

    if silhouette > best_silhouette:
        best_n_clusters = n_clusters
        best_silhouette = silhouette

kmeans = KMeans(n_clusters= best_n_clusters, random_state=42, n_init='auto')
data['cluster'] = kmeans.fit_predict(X_pca)

silhouette = silhouette_score(X_pca, data['cluster'])
print(f"Silhouette Score: {silhouette:.4f}")

# Identify underutilized spaces
underutilized_cluster = data.groupby('cluster').mean().sort_values(by=['population_density', 'existing_green_space']).index[0]
underutilized = data[data['cluster'] == underutilized_cluster]

print(f"Number of underutilized spaces identified: {len(underutilized)}")

m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=12)

# Mark all points
for _, row in data.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=4,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.5
    ).add_to(m)

# Highlight underutilized spaces
for _, row in underutilized.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=6,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.8,
        popup=f"Pop Density: {row['population_density']:.2f}, Green Space: {row['existing_green_space']:.2f}"
    ).add_to(m)

from IPython.display import display
display(m)

# Plot distribution of population density and green space
plt.figure(figsize=(8, 6))
sns.histplot(data['population_density'], kde=True, color='skyblue', label='Population Density')
sns.histplot(data['existing_green_space'], kde=True, color='lightgreen', label='Green Space')
plt.legend()
plt.title('Distribution of Population Density and Green Space')
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    data['population_density'],
    data['land_value'],
    data['existing_green_space'],
    c=data['cluster'],
    cmap='viridis',
    s=40
)
ax.set_xlabel('Population Density')
ax.set_ylabel('Land Value')
ax.set_zlabel('Green Space')
plt.colorbar(scatter)
plt.title('3D View of Clustering', fontsize=16)
plt.show()
