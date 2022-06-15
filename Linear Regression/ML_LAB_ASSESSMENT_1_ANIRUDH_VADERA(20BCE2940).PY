import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import metrics

# Importing the dataset
df=pd.read_csv("C:/Users/Anirudh/OneDrive/Desktop/weight-height.csv")
print("The dataset is as following : [10000 rows x 3 columns]")
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
print("Height : inches")
print("Weight : pounds")
print(df.info())
print("\n")

# Information related to the dataset
print("Dataset Details : ")
print(df.describe())
print("\n")

# Plot Height vs Weight............
# Blue points - Male
# Magenta points - Female
ax1 = df[df['Gender'] == 'Male'].plot(kind='scatter', x='Height', y='Weight', color='blue', alpha=0.5, figsize=(10, 7))
df[df['Gender'] == 'Female'].plot(kind='scatter', x='Height', y='Weight', color='magenta', alpha=0.5, figsize=(10 ,7), ax=ax1)
plt.legend(labels=['Males', 'Females'])
plt.title('Relationship between Height and Weight', size=24)
plt.xlabel('Height (inches)', size=18)
plt.ylabel('Weight (pounds)', size=18)

# Fitting a Regression Model
# 20% of dataset - Testing
# 80% of dataset - Training
# Random state = 123
X = df[["Height"]]
Y = df[["Weight"]]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

# Predicting the test set values
lin_pred = lin_reg.predict(X_test)

print("The X Values are : ")
print(X_test)
print("\n")

print("The predicted Y_Values are : ")
print(lin_pred)
print("\n")

# Visualizing the result in form of a scatterplot (Training Set)
plt.scatter(X_train, Y_train, color = 'Magenta')
plt.plot(X_train, lin_reg.predict(X_train), color = 'blue')
plt.legend(labels=['Height vs Weight plot(Training Set)', 'Regresion Line'])
plt.title('Relationship between Height and Weight using Regression Line(Training Set)', size=24)
plt.xlabel('Height (inches)', size=18)
plt.ylabel('Weight (pounds)', size=18)  

# Visualizing the result in form of a scatterplot (Testing Set)
plt.scatter(X_test, Y_test, color = 'Magenta')
plt.plot(X_train, lin_reg.predict(X_train), color = 'blue')
plt.legend(labels=['Height vs Weight plot(Test Set)', 'Regresion Line'])
plt.title('Relationship between Height and Weight using Regression Line(Test Set)', size=24)
plt.xlabel('Height (inches)', size=18)
plt.ylabel('Weight (pounds)', size=18)  

# Checking the accuracy of our model
print("Accuracy of the model : ")
print("\n")
print('Mean Absolute Error:', mean_absolute_error(Y_test, lin_pred))
print('Mean Squared Error:', mean_squared_error(Y_test, lin_pred))
print('Mean Root Squared Error:', np.sqrt(mean_squared_error(Y_test, lin_pred)))
print("\n")
print('Variance score: %.2f' % lin_reg.score(X_test, Y_test))
print('Coefficients: ', lin_reg.coef_)
print('R square = ',metrics.r2_score(Y_test, lin_pred)*100)

# Making new predictions based on given height
given_height = int(input("Enter the height of the student whose weight you want to predict (in inches) : "));
print("\n")
predict_weight = lin_reg.predict([[given_height]])
print("The predicted weight is (in pounds) : ", predict_weight)

# The Linear Regression Equation
print("The Coefficient is : " , lin_reg.coef_)
print("The Intercept is : " , lin_reg.intercept_)
print("The Linear regression line is : Y = " , lin_reg.intercept_[0] , "+" , lin_reg.coef_[0][0] , "*X")

# Residual Ananlysis 
plt.scatter(Y_test,lin_pred,color="Magenta")
plt.title('Checking difference between the predicted Y_Values and the original Y_Values', size=24)
plt.xlabel('Y_test set', size=18)
plt.ylabel('Y Predicted', size=18)  
