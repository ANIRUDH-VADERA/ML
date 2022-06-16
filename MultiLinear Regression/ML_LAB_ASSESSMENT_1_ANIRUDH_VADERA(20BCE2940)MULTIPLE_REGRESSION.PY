import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import seaborn as sns

# Importing the dataset
df=pd.read_csv("C:/Users/Anirudh/OneDrive/Desktop/Housing.csv")
print("The dataset is as following : [545 rows x 13 columns]")
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

# Data Preparation
# You can see that your dataset has many columns with values as 'Yes' or 'No'.
# But in order to fit a regression line, we would need numerical values and not string. 
# Hence, we need to convert them to 1s and 0s, where 1 is a 'Yes' and 0 is a 'No'.

# List of variables to map
varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Defining the map function
def binary_map(x):
    return x.map({'yes': 1, "no": 0})

# Applying the function to the housing list
df[varlist] = df[varlist].apply(binary_map)

# Dummy Variables
# The variable furnishingstatus has three levels. We need to convert these levels into integer as well.
# For this, we will use something called dummy variables.
# Get the dummy variables for the feature 'furnishingstatus' and store it in a new variable - 'status'
status = pd.get_dummies(df['furnishingstatus'])

# Now, you don't need three columns. 
# You can drop the furnished column, as the type of furnishing can be identified with just the last two columns where —

# 00 will correspond to furnished
# 01 will correspond to unfurnished
# 10 will correspond to semi-furnished

# Let's drop the first column from status df using 'drop_first = True'

status = pd.get_dummies(df['furnishingstatus'], drop_first = True)

# Add the results to the original housing dataframe
df = pd.concat([df, status], axis = 1)

# Drop 'furnishingstatus' as we have created the dummies for it
df.drop(['furnishingstatus'], axis = 1, inplace = True)

# Now let's see the head of our dataframe.
print("After Trimming and correcting the dataset looks like follows : ")
print(df.head())

# Fitting a Regression Model
# 35% of dataset - Testing
# 65% of dataset - Training
# Random state = 10
df_train, df_test = train_test_split(df,test_size=0.35,train_size=0.65,random_state=10)


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (16, 10))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()

# We can see there are some factors that affects the price of a house to great extent
# These columns are : 
# Area
# Bathrooms
# Stories
# Air conditioning
# This means changing value of even one of these factors will greatly affect the pricing of the house


# Showing the relationship between the price and the most dominant factor that is area
plt.scatter(df["area"], df["price"], color = 'Magenta')
plt.legend(labels=['Area vs Height plot'])
plt.title('Relationship between Price and area of the houses', size=24)
plt.xlabel('Area', size=18)
plt.ylabel('Price', size=18)  

# Dividing into X and Y sets for the model building
Y_train = df_train.pop('price')
X_train = df_train
# Using the Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)


Y_test = df_test.pop("price")
X_test = df_test

# Predicting the test set values
lin_pred = lin_reg.predict(X_test)

print("The X Values are : ")
print(X_test)
print("\n")

print("The predicted Y_Values are : ")
print(lin_pred)
print("\n")

print("The values look something like this : ")
X_test["Price"] = lin_pred
print(X_test)
X_test.drop(["Price"],axis=1,inplace=True)

new_lin_reg = LinearRegression()
new_lin_reg.fit(X_train["area"].values.reshape(-1,1), Y_train)

# Visualizing the result in form of a scatterplot (Training Set)
# We are only visualizing for the area factor as it is the most dominant
plt.scatter(X_train["area"], Y_train, color = 'Magenta')
plt.plot(X_train["area"], new_lin_reg.predict(X_train["area"].values.reshape(-1,1)), color = 'blue')
plt.legend(labels=['Area vs Price plot(Training Set)', 'Regresion Line'])
plt.title('Relationship between Area and Price of Houses using Regression Line(Training Set)', size=24)
plt.xlabel('Area' , size=18)
plt.ylabel('Price', size=18)  

# Visualizing the result in form of a scatterplot (Testing Set)
# We are only visualizing for the area factor as it is the most dominant
plt.scatter(X_test["area"], Y_test, color = 'Magenta')
plt.plot(X_train["area"], new_lin_reg.predict(X_train["area"].values.reshape(-1,1)), color = 'blue')
plt.legend(labels=['Area vs Price plot(Training Set)', 'Regresion Line'])
plt.title('Relationship between Area and Price of Houses using Regression Line(Test Set)', size=24)
plt.xlabel('Area' , size=18)
plt.ylabel('Price', size=18)  


# Checking the accuracy of our model
print("Accuracy of the model : ")
print("\n")
print('Mean Absolute Error:', mean_absolute_error(Y_test, lin_pred))
print('Mean Squared Error:', mean_squared_error(Y_test, lin_pred))
print('Mean Root Squared Error:', np.sqrt(mean_squared_error(Y_test, lin_pred)))
print("\n")
print('Variance score: %.2f' % lin_reg.score(X_test, Y_test))
print('R square = ',metrics.r2_score(Y_test, lin_pred)*100)

# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((Y_test - lin_pred), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label

# Making new predictions based on given features of the house
print("Enter the features of the house : ")
area = int(input("Enter the area of the house : "))
bedrooms = int(input("Enter the no of bedrooms in the house : "))
bathrooms = int(input("Enter the no of bathrooms in the house : "))
stories = int(input("Enter the stories in the house : "))
mainroad = int(input("Is there a attached mainroad 'Yes':1  'No':0 "))
guestroom = int(input("Is there a attached guestroom 'Yes':1  'No':0 "))
basement = int(input("Is there a basement 'Yes':1  'No':0 "))
hotwaterheating = int(input("Is there hot water heating 'Yes':1  'No':0 "))
airconditioning = int(input("Is there air conditioning 'Yes':1  'No':0 "))
parking = int(input("Is there parking nearby 'Yes':1  'No':0 "))
prefarea = int(input("Is there prefarea nearby 'Yes':1  'No':0 "))
furnished = int(input("Is the house semi-furnished:10 furnished:00 or unfurnihsed:01 :: "))
if (furnished==0):
    semi_furnished = 0
    unfurnished = 0
elif(furnished==1):
    semi_furnished = 0
    unfurnished = 1
else:
    semi_furnished = 1
    unfurnished = 0
predict_price = lin_reg.predict([[area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,semi_furnished,unfurnished]])
print("The predicted price of the house is : ", predict_price)

# The Linear Regression Equation
# Checking the coefficients
print("The Coefficient are : ")
coeff_df = pd.DataFrame(lin_reg.coef_,X_train.columns,columns=["Coefficient"])
print(coeff_df)
print("The Intercept is : " , lin_reg.intercept_)
print("The Linear regression line is : Price = " , lin_reg.intercept_ , "+" , lin_reg.coef_[0] , "*area" , "+" , lin_reg.coef_[1] , "*bedrooms" , "+" , lin_reg.coef_[2] , "*bathrooms" , "+" , lin_reg.coef_[3] , "*stories" , "+" , lin_reg.coef_[4] , "*mainroad" , "+" , lin_reg.coef_[5] , "*guestroom" , "+" , lin_reg.coef_[6] , "*basement" , "+" , lin_reg.coef_[7] , "*hotwaterheating" , "+" , lin_reg.coef_[8] , "*airconditioning" , "+" , lin_reg.coef_[9] , "*parking" , "+" , lin_reg.coef_[10] , "*prefarea") 

# Residual Ananlysis 
plt.scatter(Y_test,lin_pred,color="Magenta")
plt.title('Checking difference between the predicted Y_Values and the original Y_Values', size=24)
plt.xlabel('Y_test set', size=18)
plt.ylabel('Y Predicted', size=18)  

# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((Y_test - lin_pred), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)  