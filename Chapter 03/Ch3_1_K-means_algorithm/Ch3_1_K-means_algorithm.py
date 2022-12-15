import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans



data = make_blobs(n_samples = 500, n_features = 2,centers = 6, cluster_std= 1.8, random_state = 101)
plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')

model = KMeans(n_clusters=6)
model.fit(data[0])

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0], data[0][:,1], c=model.labels_, cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='rainbow')

SSE = []
list_k = list(range(1, 10))
for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(data[0])
    SSE.append(km.inertia_)

plt.figure(figsize=(6, 6))
plt.plot(list_k, SSE, '-o')
plt.xlabel('Number of clusters K')
plt.ylabel('Sum of squared distance')