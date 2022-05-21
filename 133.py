from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px

df = pd.read_csv('stars_gravity.csv')

x = df.iloc[:,[4,5]].values
WCSS = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
  kmeans.fit(x)
  WCSS.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
sns.lineplot(range(1,11), WCSS, marker='o', color="red")
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("wcss")
plt.show()