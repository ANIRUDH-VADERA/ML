import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def euclideanDistance(data_1, data_2, data_len):
    dist = 0
    for i in range(data_len):
        if(data_2[i][0] != "S" and data_2[i][0] != "P" ):
            dist = dist + np.square(float(data_1[i]) - float(data_2[i]))
    return np.sqrt(dist)

data = pd.read_csv('C:/Users/Anirudh/OneDrive/Desktop/Iris.csv', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
data=pd.DataFrame(data)
data = data.drop(labels="Id",axis=0)
print(data)

data_train,data_test=train_test_split(data,test_size=0.25)

data_train_class = np.array(data_train["class"])
data_test_class = np.array(data_test["class"])


def KNN(dataset, testInstance, k):
    distances = {}
    length = testInstance.shape[1]
    for x in range(len(dataset)):
        dist_up = euclideanDistance(testInstance, dataset.iloc[x][0:4], length)
        distances[x] = dist_up
    # Sort values based on distance
    sort_distances = sorted(distances.items(), key=operator.itemgetter(1))
    neighbors = []
    # Extracting nearest k neighbors
    for x in range(k):
        neighbors.append(sort_distances[x][0])
    # Initializing counts for 'class' labels counts as 0
    counts = {"Iris-setosa" : 0, "Iris-versicolor" : 0, "Iris-virginica" : 0}
    # Computing the most frequent class
    for x in range(len(neighbors)):
        response = dataset.iloc[neighbors[x]][-1] 
        if response in counts:
            counts[response] += 1
        else:
            counts[response] = 1
    # Sorting the class in reverse order to get the most frequest class
    sort_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    return(sort_counts[0][0])
        
# Predicting for training set

row_list = []
temp = 0
for index, rows in data_train.iterrows():
    my_list =[rows.sepal_length, rows.sepal_width, rows.petal_length, rows.petal_width]       
    if(my_list[0] != 'SepalLengthCm'):
        row_list.append([my_list])
    else:
        for i in range(len(data_train.index)):
            if(int(data_train.index[i]) == temp):
                data_train = data_train.drop(labels=i, axis=0)
    temp=temp+1
# k values for the number of neighbors that need to be considered
k_n = [1, 3, 5, 7]
# Performing kNN on the development set by iterating all of the development set data points and for each k and each distance metric

development_set_obs_k = {}
for k in k_n:
    development_set_obs = []
    for i in range(len(row_list)):
        development_set_obs.append(KNN(data_train, pd.DataFrame(row_list[i]), k))
    development_set_obs_k[k] = development_set_obs
# Nested Dictionary containing the observed class for each k and each distance metric (obs_k of the form obs_k[dist_method][k])
# print(development_set_obs_k)

print("For Training Set : ")
for key in development_set_obs_k:
    predicted = development_set_obs_k[key]
    actual = data_train_class
    print("For Key : ", key , "Accuracy Score : ", accuracy_score(actual,predicted))

# Predicting for training set

row_list = []
temp = 0
for index, rows in data_test.iterrows():
    my_list =[rows.sepal_length, rows.sepal_width, rows.petal_length, rows.petal_width]       
    if(my_list[0] != 'SepalLengthCm'):
        row_list.append([my_list])
    else:
        for i in range(len(data_test.index)):
            if(int(data_test.index[i]) == temp):
                data_test = data_test.drop(labels=i, axis=0)
    temp=temp+1
# k values for the number of neighbors that need to be considered
k_n = [1, 3, 5, 7]
# Performing kNN on the development set by iterating all of the development set data points and for each k and each distance metric

development_set_obs_k = {}
for k in k_n:
    development_set_obs = []
    for i in range(len(row_list)):
        development_set_obs.append(KNN(data_train, pd.DataFrame(row_list[i]), k))
    development_set_obs_k[k] = development_set_obs
# Nested Dictionary containing the observed class for each k and each distance metric (obs_k of the form obs_k[dist_method][k])
# print(development_set_obs_k)

print("For Test Set : ")
for key in development_set_obs_k:
    predicted = development_set_obs_k[key]
    actual = data_test_class
    print("For Key : ", key , "Accuracy Score : ", accuracy_score(actual,predicted))
    
