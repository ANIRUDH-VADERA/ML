import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

# from sklearn import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,mean_squared_error,f1_score,recall_score
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, accuracy_score, precision_score, recall_score,classification_report


df=pd.read_csv("C:/Users/Anirudh/OneDrive/Desktop/kamyr-digester.csv")
df=df.drop(["Observation"],axis=1)

y=df["Y-Kappa"]
df=df.drop(["Y-Kappa"],axis=1)

print("ANIRUDH VADERA (20BCE2940)")

print("The df is as following : ")
print(df)
print("\n")

# Check for missing values
print("Checking for missing values :")
print(df.isnull().sum())
print("\n")

# Check for NAN values
print("Number of nan values before imputations : ")
print(df.isna().sum().sum())
print("\n")


imputer=SimpleImputer()
imputedData=pd.DataFrame(imputer.fit_transform(df))
print("Number of NAN values after Simple Imputation :",imputedData.isna().sum().sum())

# Printing the header of the df
print("df Header : ")
print(imputedData.head())
print("\n")
print(imputedData)
print("\n")

# Information regarding the columns
print("Information regarding the columns : ")
print(imputedData.info())
print("\n")

scaler = preprocessing.MinMaxScaler()
names = df.columns
df_normalized = scaler.fit_transform(imputedData)
scaled_df = pd.DataFrame(df_normalized, columns=names)
scaled_df.head()
scaled_df

pca_20=PCA(n_components=int(22*0.2))
pca_20.fit(scaled_df)
df_pca_20=pca_20.transform(scaled_df)
df_pca_20.shape
pd.DataFrame(df_pca_20).head()

x=[]
for i in range(len(y)):
    x.append(i)
plt.hist(y)  
for i in (plt.hist(y)[1] ) :
    print("Percentage of Data : " ,(i/sum(plt.hist(y)[1]))*100,"%")
plt.show()

print(df_pca_20.shape)

ros = RandomOverSampler()
X, y = ros.fit_resample(df_pca_20, y.astype('int'))
sm = SMOTE(random_state=0,k_neighbors=3)
X, y = sm.fit_resample(X, y)

x=[]
for i in range(len(y)):
    x.append(i)
plt.hist(y)  
for i in (plt.hist(y)[1] ) :
    print("Percentage of Data : " ,(i/sum(plt.hist(y)[1]))*100,"%")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=36)
clf = RandomForestClassifier(max_depth=10, random_state=80)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
sc=clf.score(X_test, y_test)

# Checking the accuracy of our model
print('Accuracy: ',(accuracy_score(y_test,y_pred)*100),"%")
print('Precision: %.3f' % precision_score(y_test, y_pred,average='micro'))
print('Recall: %.3f' % recall_score(y_test, y_pred,average='micro'))
print("Mean Absolute Error:",mean_absolute_error(y_test,y_pred).round(2))
print("Mean Squared Error:",mean_squared_error(y_test,y_pred).round(2))

# Our Model Report
print('*************** Evaluation on Our Model ***************')
score_te = clf.score(X_test, y_test)
print('Accuracy Score: ', score_te*100)
# Look at classification report to evaluate the model
print(classification_report(y_test, y_pred))
print('--------------------------------------------------------')
print("")

# Additional Plots
import seaborn as sns
sns.pairplot(pd.concat([pd.DataFrame(X),pd.DataFrame(y)],axis=1))
