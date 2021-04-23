import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import distance


#save the data in a dataframe
df = pd.read_csv('galaxies.txt') 

#putting variable into a numpy array
points = df.values

#using the Kmeans instance called model, to create 3 clusters
model =KMeans(n_clusters = 3)

#the .fit  would fit the model to the array of points
model.fit(points)

#the .predict model would predict the cluster labels of points
labels = model.predict(points)
labels.shape

#a function that returns 3 numpy arrays each one with the points associated for each class
def separate_labels(labels, points):
    data_0 = []
    data_1 = []
    data_2 = [] 
    for label, point in zip(labels, points):
        if label == 0:
            data_0.append(point) #If the label is 0 they go into data_0
        elif label ==1:
            data_1.append(point) #If the label is 1 they go into data_1
        elif label ==2:
            data_2.append(point) #If the label is 2 they go into data_2
        else:
            return print("Error")
    return np.array(data_0), np.array(data_1), np.array(data_2)


g1, g2, g3 = separate_labels(labels, points)


#Ploting the different galaxies to see there respective position in space.
plt.figure(figsize=(12,8))
plt.scatter(g1[:,0], g1[:,1])
plt.scatter(g2[:,0], g2[:,1])
plt.scatter(g3[:,0], g3[:,1])
plt.legend(labels=['g1', "g2", "g3"])
#plt.show()

#ploting g1, the Galaxy where BebeYoda could be
max_coord = 0
coord_idx = 0

for i, coord in enumerate(g3[:,0]):
    if coord > max_coord:
        max_coord = coord
        coord_idx = i
print(coord_idx)
print(max_coord)


plt.figure(figsize=(12,8))
plt.scatter(g3[:,0], g3[:,1])
#plt.show()

planet_x, planet_y = g3[62, 0], g3[62,1]


#plot to identify the rightmost planet on the g1 galaxy where BeBeYoda could be
plt.figure(figsize=(12,8))
plt.scatter(g1[:,0], g1[:,1])
plt.scatter(g2[:,0], g2[:,1])
plt.scatter(g3[:,0], g3[:,1])
plt.scatter(planet_x, planet_y)
plt.legend(labels=["g1", "g2", "g3", "Yoda"])
#plt.show()


#from the planet data, extracting the force conc., and principal componets
pdf = pd.read_csv('planet.txt')


planet_pca = PCA(n_components=2)
ppc = planet_pca.fit_transform(pdf) # planet principle components is np.ndarry
df_ppc = pd.DataFrame(ppc, columns=["Principal Component 1", "Principal Component 2"])

mean_x = np.mean(ppc[:,0])
#print(f"Gravitational centre: {mean_x}, {mean_y}")
mean_y = np.mean(ppc[:,1])


def closest_node(centre, possible_locations):
    closest_index = distance.cdist([centre], possible_locations).argmin()
    return (closest_index,possible_locations[closest_index])


loc = closest_node((mean_x, mean_y), ppc)
print(loc)

print(f"Yoda located at: {round(ppc[631,0],2)}, {round(ppc[631,1],2)}")
plt.figure(figsize=(12,8))
plt.scatter(df_ppc["Principal Component 1"], df_ppc["Principal Component 2"], alpha=0.6)
plt.scatter(mean_x, mean_y)
plt.scatter(ppc[684,0], ppc[684,1])
plt.axis('equal')
plt.show()
