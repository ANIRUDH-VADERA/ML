# Importing the libraries  
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  

# Importing the dataset  
dataset = pd.read_csv('C:/Users/Anirudh/OneDrive/Desktop/Mall_Customers.csv')  
print("ANIRUDH VADERA (20BCE2940)")
print("The dataset is as following : ")
print(dataset)
print("\n")

# Check for missing values
print("Checking for missing values :")
print(dataset.isnull().sum())
print("\n")

# Printing the header of the dataset
print("Dataset Header : ")
print(dataset.head())
print("\n")

# Information regarding the columns
print("Information regarding the columns : ")
print(dataset.info())
print("\n")

# Information related to the dataset
print("Dataset Details : ")
print(dataset.describe())
print("\n")

# Choosing the variabled that is of our use
x = dataset.iloc[:, [3, 4]].values  

col1 = dataset.iloc[:,3].values
col2 = dataset.iloc[:,4].values

y_test = []
for i in range(len(dataset)):
    if(col1[i]<=60):
        y_test.append(0)
    elif(col1[i]<=130):
        if(col1[i]<=130 and col2[i]<=100 and col2[i]>60):
            y_test.append(4)
        else:
            y_test.append(3)
    elif(col1[i]<=220 and col2[i]<=80):
        y_test.append(2)
    elif(col1[i]<=300 and col2[i]<=100):
        y_test.append(1)
    else:
        y_test.append(1)
        
print(y_test)

dataset["Prediction"] = y_test

dataset.Prediction=dataset.Prediction.replace({0:"low income and mid spending", 1:"high income and high spending", 2:"mid income and mid spending", 3:"low income and low spending", 4:"low income and high spending"})


#Finding the optimal number of clusters using the dendrogram  
import scipy.cluster.hierarchy as shc  
mtp.figure(figsize=(18, 50))
dendro = shc.dendrogram(shc.linkage(x, method="ward"),leaf_rotation=0, leaf_font_size=12, orientation='right')  
mtp.title("Dendrogrma Plot")  
mtp.ylabel("Euclidean Distances")  
mtp.xlabel("Customers")  
mtp.show()  

#training the hierarchical model on dataset  
from sklearn.cluster import AgglomerativeClustering  
hc= AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
y_pred= hc.fit_predict(x)  

mtp.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s = 100, c = 'blue', label = 'Cluster 1')  
mtp.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s = 100, c = 'green', label = 'Cluster 2')  
mtp.scatter(x[y_pred== 2, 0], x[y_pred == 2, 1], s = 100, c = 'red', label = 'Cluster 3')  
mtp.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')  
mtp.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')  
mtp.title('Clusters of customers')  
mtp.xlabel('Annual Income (k$)')  
mtp.ylabel('Spending Score (1-100)')  
mtp.legend(loc='upper left')  
mtp.show()  

print("Sum of squared error: %.2f" % (sum(pow(y_pred-y_test,2))))