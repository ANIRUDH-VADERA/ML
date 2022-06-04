import matplotlib.pyplot as plt 
import pandas as pd  
from sklearn.model_selection import train_test_split    
from sklearn.tree import DecisionTreeClassifier  
from sklearn import tree
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, accuracy_score, precision_score, recall_score,classification_report
import seaborn as sns

df = pd.read_table("C:/Users/Anirudh/OneDrive/Desktop/fruit_data_with_colors.txt")
df = pd.DataFrame(df)

print("The dataset is as following :")
print(df)
print("\n")

# Check for missing values
print("Checking for missing values :")
print(df.isnull().sum())
print("\n")

# Printing the header of the dataset
print("Dataset Header : ")
print(df.head())
print("\n")

# Information regarding the columns
print("Information regarding the columns : ")
print(df.info())
print("\n")

# Information related to the dataset
print("Dataset Details : ")
print(df.describe())
print("\n")

# correlation matrix
sns.heatmap(df.corr())

# Dummy Variables
# The variable fruit_subtype has many levels. We need to convert these levels into integer as well in order to predict
# For this, we will use something called dummy variables.
# Get the dummy variables for the feature 'fruit_subtype' and store it in a new variable - 'status'
status = pd.get_dummies(df['fruit_subtype'], drop_first = True)

# Now, you don't need all the columns. 
# You can drop the fruit_subtype column, as the fruit_subtype can be identified with just the last 8 columns where encoding has already been done
# Add the results to the original dataframe
df = pd.concat([df, status], axis = 1)

# Drop 'fruit_subtype' as we have created the dummies for it
df.drop(['fruit_subtype'], axis = 1, inplace = True)

# Now let's see the head of our dataframe.
print("After Trimming and correcting the dataset looks like follows : ")
print(df.head())

# Extracting Independent and dependent Variable  
X = df.iloc[:, 2:14].values  
Y = df.iloc[:, 0].values  

# Splitting the dataset into training and testing set
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.36, random_state=10)  

#Fitting Decision Tree classifier to the training set  
classifier= DecisionTreeClassifier(criterion='gini', random_state=0)  
classifier.fit(X_train, Y_train)  

#Predicting the test set result  
Y_pred = classifier.predict(X_test)  

#Creating the Confusion matrix  
c = confusion_matrix(Y_test,Y_pred)
print(c)

class_names = ["Aple","Mandarin","Orange","Lemon"]
sns.heatmap(c, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=class_names, yticklabels=class_names)
#Plotting the Confusion Matrix
plt.ylabel('Actual Label', fontsize=18)
plt.xlabel('Predicted Label', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# Checking the accuracy of our model
print('Accuracy: ',accuracy_score(Y_test,Y_pred))
print('Precision: %.3f' % precision_score(Y_test, Y_pred,average='micro'))
print('Recall: %.3f' % recall_score(Y_test, Y_pred,average='micro'))

# The decision tree
print(tree.plot_tree(classifier,filled=True,precision = 4))

# Our Model Report
print('*************** Evaluation on Our Model ***************')
score_te = classifier.score(X_test, Y_test)
print('Accuracy Score: ', score_te)
# Look at classification report to evaluate the model
print(classification_report(Y_test, Y_pred))
print('--------------------------------------------------------')
print("")


# Pre pruning

max_depth = []
acc_gini = []
acc_entropy = []
var = []
for i in range(1,6):
  
    dtree = DecisionTreeClassifier(criterion='gini', random_state=0)
    dtree.fit(X_train, Y_train)
    pred = dtree.predict(X_test)
    var.append(accuracy_score(Y_test, pred))
      
    dtree = DecisionTreeClassifier(criterion='gini', max_depth=i)
    dtree.fit(X_train, Y_train)
    pred = dtree.predict(X_test)
    acc_gini.append(accuracy_score(Y_test, pred))
    
    dtree = DecisionTreeClassifier(criterion='entropy', max_depth=i)
    dtree.fit(X_train, Y_train)
    pred = dtree.predict(X_test)
    acc_entropy.append(accuracy_score(Y_test, pred))
    
    max_depth.append(i)

d = pd.DataFrame({'acc_gini':pd.Series(acc_gini), 
  'acc_entropy':pd.Series(acc_entropy),
  'max_depth':pd.Series(max_depth),
  'var':pd.Series(var)
  })
# visualizing changes in parameters
plt.plot('max_depth','var', data=d, label='pre pruned tree')
plt.plot('max_depth','acc_gini', data=d, label='gini')
plt.plot('max_depth','acc_entropy', data=d, label='entropy')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.legend()

dtree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
dtree.fit(X_train, Y_train)
pred = dtree.predict(X_test)


plt.plot('max_depth','var', data=d, label='pre pruned tree')
plt.plot('max_depth','acc_entropy', data=d, label='entropy')
plt.plot('max_depth','acc_gini', data=d, label='gini')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.legend()

# The decision tree
print(tree.plot_tree(dtree,filled=True,precision = 4))


#Creating the Confusion matrix  
c = confusion_matrix(Y_test,pred)
print(c)

class_names = ["Aple","Mandarin","Orange","Lemon"]
sns.heatmap(c, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=class_names, yticklabels=class_names)
#Plotting the Confusion Matrix
plt.ylabel('Actual Label', fontsize=18)
plt.xlabel('Predicted Label', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# Checking the accuracy of our model
print('Accuracy: ',accuracy_score(Y_test,pred))
print('Precision: %.3f' % precision_score(Y_test, pred,average='micro'))
print('Recall: %.3f' % recall_score(Y_test, pred,average='micro'))

# Our Model Report
print('*************** Evaluation on Our Model ***************')
score_te = dtree.score(X_test, Y_test)
print('Accuracy Score: ', score_te)
# Look at classification report to evaluate the model
print(classification_report(Y_test, pred))
print('--------------------------------------------------------')
print("")
