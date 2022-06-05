import pandas as pd
import random
import math

dataset = pd.read_csv("C:/Users/Anirudh/OneDrive/Desktop/IRIS.csv");

k = int(input("Enter the value of k : "))

X = dataset.iloc[:,0:4]
Y = dataset.iloc[:,4]

euclidean_dis = []
c_index = []

for i in range(k):
    euclidean_dis.append([])
    c_index.append(random.randint(0,len(dataset)-1))
    
change = 1
class_given = []
f=0
old=[]

while(change==1):
    if(f==1):      
        old = class_given
        class_given = []
        euclidean_dis = []
        for i in range(k):
            euclidean_dis.append([])
    for i in range(k):
        if(f==0):
            centroid = X.iloc[c_index[i],:]
        else:
            centroid = c_index[i]
        for j in range(len(X)):
            w = X.iloc[j,0]
            x = X.iloc[j,1]
            y = X.iloc[j,2]
            z = X.iloc[j,3]
            e = math.sqrt(math.pow((centroid[0]-w),2)+math.pow((centroid[1]-x),2)+math.pow((centroid[2]-y),2)+math.pow((centroid[3]-z),2))
            euclidean_dis[i].append(e)
    f=1
    temp = []
    for i in range(len(X)):
        temp = []
        for j in range(k):
            temp.append(euclidean_dis[j][i])
        min_idx = 0
        for m in range(len(temp)):
            if(temp[m]<=temp[min_idx]):
                min_idx=m
        class_given.append(min_idx)
    c_index = []
    for i in range(k):
        nw = 0
        nx = 0
        ny = 0
        nz = 0
        t = []
        for j in range(len(class_given)):
            if(class_given[j]==i):
                t.append(X.iloc[j,:])
        for m in t:
            nw = nw + m[0]
            nx = nx + m[1]
            ny = ny + m[2]
            nz = nz + m[3]
        nw = nw/(len(t))
        nx = nx/(len(t))
        ny = ny/(len(t))
        nz = nz/(len(t))
        c_index.append([nw,nx,ny,nz])
    if(old==class_given):
        change = 0

print(class_given)


