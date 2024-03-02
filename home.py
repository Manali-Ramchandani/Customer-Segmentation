import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Sidebar - Choose the number of clusters
st.sidebar.header('User Input Parameters')
number_of_clusters = st.sidebar.slider('Number of clusters', 2, 10, 5)


# Main header
st.title('Customer Segmentation App')

with st.expander("See Project Description"):
    st.success("""
    - In the fast-paced retail sector, effectively understanding and segmenting customers is crucial for targeted marketing strategies.
    - This project utilizes the K-Means Clustering Algorithm, an unsupervised machine learning technique, to identify distinct customer groups within a mall environment. 
    - Our aim is to reveal actionable insights that enable the marketing team to devise tailored campaigns, enhancing customer engagement and loyalty. Hosted on Streamlit, 
            this application provides an intuitive platform for exploring customer segmentation, making sophisticated data analysis accessible to all stakeholders. 
    - By pinpointing key customer segments, we equip our marketing team with the knowledge to focus their efforts more efficiently, optimizing marketing spend and driving sales.
    - This project not only showcases the potential of machine learning in retail but also demonstrates how data-driven insights can be transformed into strategic marketing decisions, all through an interactive and user-friendly interface.""")

# Display the dataset
data = pd.read_csv('data/Mall_Customers.csv')
st.subheader('1. Looking at our data')
st.write(data)
# Run K-Means and calculate silhouette score
X = data.iloc[:, [3, 4]].values # Assuming these are your clustering features
kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Display clusters
st.subheader('2. Displaying Clusters')
fig, ax = plt.subplots()
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=kmeans.labels_, palette='viridis', ax=ax)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('Customer Groups')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
st.pyplot(fig)

# Display correlation matrix
corr_matrix = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].corr()
st.subheader('3. Correlation Matrix')
fig, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, ax=ax)
st.pyplot(fig)


st.sidebar.warning("by Manali Ramchandani...")