import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
# %matplotlib inline
# -----DATASET ENCODING-----
dataset = pd.read_csv('/home/prinzz/Desktop/Work/Data Science/Assignment/Python/car data.csv')
car_name_integer = LabelEncoder().fit_transform(dataset["Car_Name"])
fuel_type_integer = LabelEncoder().fit_transform(dataset["Fuel_Type"])
seller_type_integer = LabelEncoder().fit_transform(dataset["Seller_Type"])
trans_integer = LabelEncoder().fit_transform(dataset["Transmission"])
dataset.Car_Name = car_name_integer
dataset.Fuel_Type = fuel_type_integer
dataset.Seller_Type = seller_type_integer
dataset.Transmission = trans_integer
# -----DATASET ENCODING-----

X =pd.DataFrame(dataset[['Car_Name', 'Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']].values)
y = dataset['Selling_Price'].values

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['Selling_Price'])
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
# print(df1)
# -----PREDICTED Graph
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# print(coeff_df)
