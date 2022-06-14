import matplotlib.pyplot as plt 
import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split    
from sklearn.tree import DecisionTreeClassifier  
from sklearn import tree
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, accuracy_score, precision_score, recall_score,classification_report
import seaborn as sns
from sklearn.neural_network import MLPClassifier

# Importing the dfset
df=pd.read_csv("C:/Users/Anirudh/OneDrive/Desktop/car_evaluation.csv")
print(df)
print("\n")

# Check for missing values
print("Checking for missing values :")
print(df.isnull().sum())
print("\n")

# Printing the header of the dfset
print("dfset Header : ")
print(df.head())
print("\n")

# Information regarding the columns
print("Information regarding the columns : ")
print(df.info())
print("\n")

# Information related to the dfset
print("dfset Details : ")
print(df.describe())
print("\n")

# Convert categories into integers for each column.
df.Buying=df.Buying.replace({'low':0, 'med':1, 'high':2, 'vhigh':3})
df.Maint=df.Maint.replace({'low':0, 'med':1, 'high':2, 'vhigh':3})
df.Doors=df.Doors.replace({'2':0, '3':1, '4':2, '5more':3})
df.Persons=df.Persons.replace({'2':0, '4':1, 'more':2})
df.Lug_Boot=df.Lug_Boot.replace({'small':0, 'med':1, 'big':2})
df.Safety=df.Safety.replace({'low':0, 'med':1, 'high':2})
df.Decision=df.Decision.replace({'unacc':0, 'acc':1, 'good':2, 'vgood':3})

# Now let's see the head of our dfframe.
print("After Trimming and correcting the dfset looks like follows : ")
print(df.head())

# Extracting Independent and dependent Variable  
X = df[df.columns[:-1]]
Y = df['Decision']

# Splitting the dfset into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.1, random_state=10)  

# Initialize a Multi-layer Perceptron classifier.

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,random_state=10, max_iter=1000, shuffle=True, verbose=False)

# Train the classifier.
mlp.fit(X_train, Y_train)

# Make predictions.
Y_pred = mlp.predict(X_test)


class_names = ["Unacc","Acc","Good","VGood"]
# Plot confusion matrix for MLP.
mlp_matrix = confusion_matrix(Y_test,Y_pred)
plt.figure(figsize=(8,8))
sns.set(font_scale=1.4)
sns.heatmap(mlp_matrix,annot=True,square=True, cbar=False,linewidth=0.5,fmt="d",cmap="Blues",xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix for Analysis');


# Actual and Predicted Values
print("Actual Values")
print(Y_test)

print("Predicted Values")
print(Y_pred)

print("Prediction of first ten elements : ",list(Y_pred)[0:10])
print("Actual Value of first ten elements: ",list(Y_test)[0:10])

# Checking the accuracy of our model
print('Accuracy: ',accuracy_score(Y_test,Y_pred))
print('Precision: %.3f' % precision_score(Y_test, Y_pred,average='micro'))
print('Recall: %.3f' % recall_score(Y_test, Y_pred,average='micro'))

# Our Model Report
print('*************** Evaluation on Our Model ***************')
print('Accuracy Score: ', accuracy_score(Y_test,Y_pred))
# Look at classification report to evaluate the model
print(classification_report(Y_test, Y_pred))
print('--------------------------------------------------------')
print("")


print('*************** Weights and Bias ***************')

print("weights between input and first hidden layer:")
print(mlp.coefs_[0])
print("\nweights between first hidden and second hidden layer:")
print(mlp.coefs_[1])


print("We can generalize the above to access a neuron in the following way:")
print(mlp.coefs_[0][:,0])

print("w0 = ", mlp.coefs_[0][0][0])
print("w1 = ", mlp.coefs_[0][1][0])
print("w2 = ", mlp.coefs_[0][2][0])
print("w3 = ", mlp.coefs_[0][3][0])
print("w4 = ", mlp.coefs_[0][4][0])
print("w5 = ", mlp.coefs_[0][5][0])

    
print("Bias values for first hidden layer:")
print(mlp.intercepts_[0])
print("\nBias values for second hidden layer:")
print(mlp.intercepts_[1])
