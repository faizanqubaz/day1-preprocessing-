# By Suneet Jain
# Importing the libaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data=pd.read_csv(r"C:/Users/CL/Desktop/Data.csv")

# we have only 9 rows data and there are two missing fields we dont delete or replace it becasue the data is small hence we replace it with the data average values
# with the help of simpleimputer function
from sklearn.impute import SimpleImputer

# 'np.nan' signifies that we are targeting missing values
# and the strategy we are choosing is replacing it with 'mean'

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(data.iloc[:, 1:3])
data.iloc[:, 1:3] = imputer.transform(data.iloc[:, 1:3]) 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# [0] signifies the index of the column we are appliying the encoding on
data = pd.DataFrame(ct.fit_transform(data))


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data.iloc[:,-1] = le.fit_transform(data.iloc[:,-1])
# 'data.iloc[:,-1]' is used to select the column that we need to be encoded

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(data))

# Step 6: Splitting the dataset

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
# .values function coverts the data into arrays
print("Independent Variable\n")
print(X)
print("\nDependent Variable\n")
print(y)

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#'test_size=0.2' means 20% test data and 80% train data
print(X_test)
print(X_train)