import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


#helper 
def get_index_from_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None  # Value not found



data = pd.read_csv('data/Mall_Customers.csv')

st.subheader("Getting the ideal number of clusters with Silhouette Scores", divider='rainbow') 

X = data.iloc[:, [3, 4]].values


# Silhouette scores for different numbers of clusters
silhouette_scores = {}

for i in range(2, 11):  # Starting from 2 clusters since silhouette score cannot be calculated for a single cluster
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores[i] = score


# Plotting the silhouette scores
fig = plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), list(silhouette_scores.values()), marker='o')
plt.title('Silhouette Score Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()
st.pyplot(fig)


ideal_cluster_number = get_index_from_value(silhouette_scores, max((silhouette_scores).values()))
st.success(f'The ideal cluster number is {ideal_cluster_number} with a silhouette score of {round(max((silhouette_scores).values()),2)}')
