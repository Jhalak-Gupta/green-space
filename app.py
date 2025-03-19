from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import folium
from folium.plugins import MarkerCluster

app = Flask(__name__)

# Load Data and Train Model
data = pd.read_csv('city_data.csv')

# Normalize data
scaler = MinMaxScaler()
features = data[['population_density', 'land_value', 'existing_green_space']]
scaled_features = scaler.fit_transform(features)

# PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled_features)

# KMeans clustering
best_n_clusters = 4
kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init='auto')
data['cluster'] = kmeans.fit_predict(X_pca)

# Generate map
def generate_map():
    city_map = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=12)

    # MarkerCluster groups the markers together when zoomed out
    marker_cluster = MarkerCluster().add_to(city_map)

    # Add data points
    for _, row in data.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Cluster: {row['cluster']}, Density: {row['population_density']}",
            icon=folium.Icon(color='green' if row['cluster'] == 1 else 'blue' if row['cluster'] == 2 else 'red')
        ).add_to(marker_cluster)

    # Save to HTML
    city_map.save('static/map.html')

# Flask Routes
@app.route('/')
def home():
    generate_map()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        population_density = float(request.form['population_density'])
        land_value = float(request.form['land_value'])
        existing_green_space = float(request.form['existing_green_space'])

        # Create input array
        input_data = np.array([[population_density, land_value, existing_green_space]])
        input_scaled = scaler.transform(input_data)
        input_pca = pca.transform(input_scaled)

        # Predict cluster
        cluster = kmeans.predict(input_pca)[0]
        
        result = {
            'cluster': int(cluster),
            'message': f'This area belongs to Cluster {cluster}'
        }

        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/map')
def show_map():
    return render_template('map.html')

if __name__ == "__main__":
    app.run(debug=True)
