import pandas as pd
# reading data
data = pd.read_csv("delaney_solubility_with_descriptors.csv")
# data preparation
y = data['logS']
X = data.drop("logS", axis=1)
# data splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=100)

# Model building
# Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
# applying the model for prediction
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

# Model Performance
from sklearn.metrics import mean_squared_error, r2_score
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train,y_lr_train_pred)
lr_test_mse = mean_squared_error(y_test,y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred )

# making a dataframe from the results
result = pd.DataFrame(["Linear Regression", lr_test_mse, lr_test_r2, lr_train_mse, lr_train_r2]).transpose()
result.columns =["Method", "Training MSE", "Training R2", "Test MSE", "Test R2"]
# print(result)

# Random Forest if LogS is quantitative else category
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

# applying the model for prediction
y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

# model performance
from sklearn.metrics import mean_squared_error, r2_score
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train,y_rf_train_pred)
rf_test_mse = mean_squared_error(y_test,y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred )


rf_result = pd.DataFrame(["Forrest Regression", rf_test_mse, rf_test_r2, rf_train_mse, rf_train_r2]).transpose()
rf_result.columns =["Method", "Training MSE", "Training R2", "Test MSE", "Test R2"]
# print(rf_result)
all_result = pd.concat([result,rf_result], axis=0).reset_index(drop=True)

# Data visualization
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(5,5))
plt.scatter(x=y_train,y=y_lr_train_pred,c="Green", alpha=0.3)
# Make a trend line
z = np.polyfit(y_train,y_lr_train_pred,1)
p = np.poly1d(z)
plt.plot(y_train,p(y_train),"Red")
# plot labels
plt.ylabel("Predicted LogS")
plt.xlabel("Experimental LogS")
plt.show()

