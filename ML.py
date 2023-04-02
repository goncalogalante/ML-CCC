import pandas as pd
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# function to validation
def validate_regression(x_train, y_train, alpha_range, cv_size):

    # Splitting data into training and test sets
    X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_val)
    lr_mse = mean_squared_error(y_val, lr_pred)
    
    # Ridge Regression
    ridge = Ridge()
    ridge_params = {'alpha': alpha_range}
    ridge_cv = GridSearchCV(ridge, param_grid=ridge_params, cv=cv_size)
    ridge_cv.fit(X_train, y_train)
    ridge_pred = ridge_cv.predict(X_val)
    ridge_mse = mean_squared_error(y_val, ridge_pred)
    
    # Lasso Regression
    lasso = Lasso()
    lasso_params = {'alpha': alpha_range}
    lasso_cv = GridSearchCV(lasso, param_grid=lasso_params, cv=cv_size)
    lasso_cv.fit(X_train, y_train)
    lasso_pred = lasso_cv.predict(X_val)
    lasso_mse = mean_squared_error(y_val, lasso_pred)
    
    
    # ElasticNet Regression
    from sklearn.linear_model import ElasticNet
    enet = ElasticNet()
    enet_params = {'alpha': alpha_range, 'l1_ratio': np.arange(0.1, 1.0, 0.1)}
    enet_cv = GridSearchCV(enet, param_grid=enet_params, cv=cv_size)
    enet_cv.fit(X_train, y_train)
    enet_pred = enet_cv.predict(X_val)
    enet_mse = mean_squared_error(y_val, enet_pred)
    
    # Support Vector Regression
    from sklearn.svm import SVR
    svr = SVR(kernel='linear')
    svr_params = {'C': alpha_range}
    svr_cv = GridSearchCV(svr, param_grid=svr_params, cv=cv_size)
    svr_cv.fit(X_train, y_train)
    svr_pred = svr_cv.predict(X_val)
    svr_mse = mean_squared_error(y_val, svr_pred)
    
    # Random Forest Regression
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(random_state=42)
    rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]}
    rf_cv = GridSearchCV(rf, param_grid=rf_params, cv=cv_size)
    rf_cv.fit(X_train, y_train)
    rf_pred = rf_cv.predict(X_val)
    rf_mse = mean_squared_error(y_val, rf_pred)
    
    # Gradient Boosting Regression
    from sklearn.ensemble import GradientBoostingRegressor
    gbr = GradientBoostingRegressor(random_state=42)
    gbr_params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]}
    gbr_cv = GridSearchCV(gbr, param_grid=gbr_params, cv=cv_size)
    gbr_cv.fit(X_train, y_train)
    gbr_pred = gbr_cv.predict(X_val)
    gbr_mse = mean_squared_error(y_val, gbr_pred)
    
    # AdaBoost Regression
    from sklearn.ensemble import AdaBoostRegressor
    ada = AdaBoostRegressor(random_state=42)
    ada_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}
    ada_cv = GridSearchCV(ada, param_grid=ada_params, cv=cv_size)
    ada_cv.fit(X_train, y_train)
    ada_pred = ada_cv.predict(X_val)
    ada_mse = mean_squared_error(y_val, ada_pred)
    
     # XGBoost Regression
    from xgboost import XGBRegressor
    xgb = XGBRegressor(random_state=42)
    xgb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [3, 5, 7]}
    xgb_cv = GridSearchCV(xgb, param_grid=xgb_params, cv=cv_size)
    xgb_cv.fit(X_train, y_train)
    xgb_pred = xgb_cv.predict(X_val)
    xgb_mse = mean_squared_error(y_val, xgb_pred)
    
    print("Linear Regression")
    print("MSE:", lr_mse)
    print("RMSE:", np.sqrt(lr_mse))
    print("------------")
    print("Ridge Regression")
    print("Best alpha:", ridge_cv.best_params_['alpha'])
    print("MSE:", ridge_mse)
    print("RMSE:", np.sqrt(ridge_mse))
    print("------------")
    print("Lasso Regression")
    print("Best alpha:", lasso_cv.best_params_['alpha'])
    print("MSE:", lasso_mse)
    print("RMSE:", np.sqrt(lasso_mse))
    print("------------")
    print("ElasticNet Regression")
    print("Best parameters: ", enet_cv.best_params_)
    print("MSE: ", enet_mse)
    print("RMSE:", np.sqrt(enet_mse))
    print("------------")
    print("Support Vector Regression")
    print("Best parameters: ", svr_cv.best_params_)
    print("MSE: ", svr_mse)
    print("RMSE:", np.sqrt(svr_mse))
    print("------------")
    print("Random Forest Regression")
    print("Best parameters: ", rf_cv.best_params_)
    print("MSE: ", rf_mse)
    print("RMSE:", np.sqrt(rf_mse))
    print("------------")
    print("Gradient Boosting Regression")
    print("Best parameters:", gbr_cv.best_params_)
    print("MSE:", gbr_mse)
    print("RMSE:", np.sqrt(gbr_mse))
    print("------------")
    print("AdaBoost Regression")
    print("Best parameters:", ada_cv.best_params_)
    print("MSE:", ada_mse)
    print("RMSE:", np.sqrt(ada_mse))
    print("------------")
    print("XGBoost Regression")
    print("Best parameters:", xgb_cv.best_params_)
    #print("R2 score:", xgb_r2)
    print("MSE:", xgb_mse)
    print("RMSE:", np.sqrt(xgb_mse))
    print("------------")
 
#--------- DATA TREATMENT ------------

# read an excel file in xlsx format
#data_df= pd.read_excel('train_data.xlsx')

# read an excel file in csv format
df_train = pd.read_csv("train_data.csv")
df_test = pd.read_csv("test_data.csv")

# change values in the whole column
#df_train['VOLTS'] = df_train['VOLTS'] + 12

# drop features
#df_train = df_train.drop("TEMP", axis=1)

# drop all the NaN variables
df_train.dropna(inplace=True) 

# drop a row under a "condition" or more
#df_train = df_train[df_train['UNIT'] != '?']
#df_train = df_train[(df_train['UNIT'] != 'K') & (df_train['UNIT'] != '?')]

# max and min values in features
min_C = df_train.loc[df_train['UNIT'] == 'C', 'TEMP'].min()
max_C = df_train.loc[df_train['UNIT'] == 'C', 'TEMP'].max()

min_K = df_train.loc[df_train['UNIT'] == 'K', 'TEMP'].min()
max_K = df_train.loc[df_train['UNIT'] == 'K', 'TEMP'].max()

count_C_unk = ((df_train['UNIT'] == '?') & (df_train['TEMP'] >= min_C) & (df_train['TEMP'] <= max_C)).sum()
count_K_unk = ((df_train['UNIT'] == '?') & (df_train['TEMP'] >= min_C) & (df_train['TEMP'] <= max_C)).sum()

print("Celsius min:", min_C, "Celsius max:", max_C, "| Kelvin min:", min_K, "Kelvin max", max_K)
print("Celsius:", count_C_unk, "? Kelvin:", count_K_unk)

# change feature values based on conditions
df_train.loc[(df_train['UNIT'] == '?') & (df_train['TEMP'] >= min_C) & (df_train['TEMP'] <= max_C), 'UNIT'] = 'C'
df_train.loc[(df_train['UNIT'] == '?') & (df_train['TEMP'] >= min_K) & (df_train['TEMP'] <= max_K), 'UNIT'] = 'K'

# change values under a "condition"
df_train.loc[df_train['UNIT'] == 'K', 'TEMP'] = df_train.loc[df_train['UNIT'] == 'K', 'TEMP'] - 273.15

# count the values in a feature
count = (df_train["UNIT"] == '?').sum()
print("counter %d",count)

# replace a value for another in a column
df_train['UNIT'] = df_train['UNIT'].replace('K', 'C')

# define the xtrain, ytrain and xtest in df mode
x_train_df = df_train.drop("OUTPUT", axis=1)
y_train_df = df_train["OUTPUT"]

x_test_df = df_test

# convert strings in numerical data
unit = {'K': 0, 'C': 1, '?':2}
power = {'high': 1, 'low': 0}
mode = {'auto': 0, 'beam': 1, 'burst':2, 'REDACTED':3}

# Replace the values in the 'POWER' column with the numerical values (train and test data)
x_train_df['UNIT'] = x_train_df['UNIT'].replace(unit)
x_train_df['POWER'] = x_train_df['POWER'].replace(power)
x_train_df['MODE'] = x_train_df['MODE'].replace(mode)

x_test_df['UNIT'] = x_test_df['UNIT'].replace(unit)
x_test_df['POWER'] = x_test_df['POWER'].replace(power)
x_test_df['MODE'] = x_test_df['MODE'].replace(mode)

print(x_train_df)
print(x_test_df)

#--------- DATA PREPROCESSING ----------

## Normalization

# training data
scaler = MinMaxScaler()
scaler.fit(x_train_df)
x_train_df_norm = scaler.transform(x_train_df)
x_train_df = pd.DataFrame(x_train_df_norm, columns=x_train_df.columns)

# test data
scaler_test = MinMaxScaler()
scaler_test.fit(x_test_df)
x_test_df_norm = scaler_test.transform(x_test_df)
x_test_df = pd.DataFrame(x_test_df_norm, columns=x_test_df.columns)

## Standardization

# training data
scaler = StandardScaler()
x_train_df = scaler.fit_transform(x_train_df)

# test data
scaler_test = StandardScaler()
x_test_df = scaler_test.fit_transform(x_test_df)

#--------- VALIDATION ----------

## Lazy 
# Splitting data into training and test sets to use in LAZY 
X_train, X_val, y_train, y_val = train_test_split(x_train_df, y_train_df, test_size=0.2, random_state=42)

# use LazyRegressor
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=mean_squared_error)
models, prediction = reg.fit(X_train, X_val, y_train, y_val)

# use LazyClassifier
#clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
#models, prediction = reg.fit(X_train, X_val, y_train, y_val)

# validation results
print(models)

## Manual Validation
alpha_range=[0.001, 0.01, 0.1, 1, 10, 100, 1000]
cv_size =  10
#val_results = validate_regression(x_train_df, y_train_df,alpha_range,cv_size)

#--------- FINAL ----------

# choose model
model = LinearRegression()
model.fit(x_train_df, y_train_df)

# predict
y_test = model.predict(x_test_df)

# Output CCC
sys.stdout = open("Solution.txt", "w")
for i in y_test:
    print(i)
    
sys.stdout.close()


