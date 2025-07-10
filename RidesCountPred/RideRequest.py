import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'c:\Users\HP\Downloads\ola.csv')
# print(df)

# pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None) 

# print(df.info())

feature = ['humidity','temp','windspeed']
for i in feature:
    df[i] = df[i].astype(int)

# print(df.info())

# print(df.describe())

# print(df.isnull().sum())
# print(df['datetime'])
# Now, I want to seperate the datetime column
# convert the datetime column into datetime datatype
df['datetime'] = pd.to_datetime(df['datetime'])

# Now, extract the date, time, year from the datetime column

df['date'] = df['datetime'].dt.strftime('%Y%m%d').astype(int)
df['month'] = df['datetime'].dt.month.astype(int)
df['year'] = df['datetime'].dt.year.astype(int)
df['time'] = df['datetime'].dt.hour.astype(int)
df['weekday'] = df['datetime'].dt.strftime('%A')
df['am_pm'] = df['datetime'].dt.strftime('%p')
df['am_pm'] = df['am_pm'].map({'AM':0,'PM':1})
# print(df['am_pm'])
# print(df.head(50))
# print(df.info())

# map the weekday column
weekday_mapping = {day: 0 if i < 5 else 1 for i, day in enumerate(df['weekday'].unique())}
df['weekday'] = df['weekday'].map(weekday_mapping)
# print(df['weekday'].head(150))

import holidays

india_holidays = holidays.India(years=df['datetime'].dt.year.unique())

df['is_holiday'] = df['datetime'].apply(lambda x: 1 if x in india_holidays else 0)
# print(df['is_holiday'])
# to print the holiday name
df['holiday_name'] = df['datetime'].apply(lambda x: india_holidays.get(x) if x in india_holidays else "None")
# print(df['holiday_name'])


# Now, remove the columns datetime and date

df1 = df.drop(['datetime','date'],axis = 1)
# print(df1)

# print(df.isnull().sum())

# If I check the outliers for every columns
# sns.boxplot(df['season']) 
# No outlier in this
# sns.boxplot(df['weather'])
# no outlier in this 
# sns.boxplot(df['temp'])
# no outlier in this 
# sns.boxplot(df['humidity'])
# sns.boxplot(df['casual'])
# sns.boxplot(df['windspeed'])
# sns.boxplot(df['registered'])

# no outlier in this

# sns.boxplot(df['holiday_name'])


# plt.show()

# To check the highly correlated features
df1_numeric = df1.apply(pd.to_numeric,errors = 'coerce')
sns.heatmap(df1_numeric.corr()>0.8,annot=True, cmap='coolwarm') 
# plt.show()

# By this heatmap, we can see that the correlation features are registered, time and count. But we cannot delete the target variable

# drop the correlation features
df2 = df1_numeric.drop(['registered','time','holiday_name','year'],axis = 1)
# print(df2)
# print(df2.corr())

# Now, train the model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

# divide the dataset into dependent and independent features

x = df2.drop(['count'],axis = 1)
y = df2['count']
# print(x)
# print(y)
x_train,x_test,y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)

# for i in range(40,50):
#     x_train,x_test,y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=i)
#     sts = StandardScaler()
#     x_train = sts.fit_transform(x_train)
#     x_test = sts.transform(x_test)
# 
#     # Now, apply the linear regression model
#     lr = LinearRegression()
#     lr.fit(x_train,y_train)
# 
#     y_train_pred = lr.predict(x_train)
#     training_accuracy_lr = r2_score(y_train, y_train_pred)
# 
#     y_test_pred = lr.predict(x_test)
#     test_accuracy_lr = r2_score(y_test, y_test_pred)
# 
#     # print({'training_accuracy': training_accuracy_lr, 'test_accuracy': test_accuracy_lr})
# 
# rdr = RandomForestRegressor(random_state=42)
# parameters = {
#     'n_estimators': [50,100,150,200],
#     'max_depth': [2,3,5],
#     
# }
# grid_rdr = GridSearchCV(rdr,parameters,cv = 5)
# grid_rdr.fit(x_train,y_train)
# # print(grid_rdr.best_params_)
# y_train_pred_rdr = grid_rdr.predict(x_train)
# training_accuracy_rdr = r2_score(y_train,y_train_pred_rdr)
# 
# y_test_pred_rdr = grid_rdr.predict(x_test)
# test_accuracy_rdr = r2_score(y_test,y_test_pred_rdr)
# # print({'training_accuracy': training_accuracy_rdr, 'test_accuracy': test_accuracy_rdr})
# 
# # If I apply the gradient boosting regressor
# 
# gbr = GradientBoostingRegressor(random_state=42)
# parameters = {
#     'n_estimators':[50,100,150,200],
#     'max_depth':[2,3,5],
#     'learning_rate':[0.01,0.1,0.2,0.5]
#     }
# 
# grid_gbr = GridSearchCV(gbr, parameters,cv = 5)
# grid_gbr.fit(x_train,y_train)
# 
# # to check the training accuracy
# y_train_pred_gbr = grid_gbr.predict(x_train)
# training_accuracy_gbr = r2_score(y_train,y_train_pred_gbr)
# 
# y_test_pred_gbr = grid_gbr.predict(x_test)
# test_accuracy_gbr = r2_score(y_test, y_test_pred_gbr)

# print({'training_accuracy_gbr': training_accuracy_gbr, 'test_accuracy_gbr': test_accuracy_gbr})

sts = StandardScaler()
x_trans_train = sts.fit_transform(x_train)
x_trans_test = sts.transform(x_test)
# print(x_trans_train)
# print(x_trans_test)

xgb = XGBRegressor(random_state=42)
parameters = {
    'n_estimators':[50,100,150,200],
    'max_depth':[2,3,5],
    'learning_rate':[0.01,0.1,0.2,0.5]
    }

grid_xgb = GridSearchCV(xgb, parameters,cv = 5)
grid_xgb.fit(x_trans_train,y_train)
# print(grid_xgb.best_params_)
# {'learning_rate': 0.01, 'max_depth': 2, 'n_estimators': 200}

xgb = XGBRegressor(learning_rate= 0.01, max_depth= 2, n_estimators= 200,random_state=42)
# to check the training accuracy
xgb.fit(x_trans_train,y_train)

y_train_pred_xgb = xgb.predict(x_trans_train)
training_accuracy_xgb = r2_score(y_train,y_train_pred_xgb)

y_test_pred_xgb = xgb.predict(x_trans_test)
test_accuracy_xgb = r2_score(y_test,y_test_pred_xgb)

# print({'training_accuracy_xgb': training_accuracy_xgb, 'test_accuracy_xgb': test_accuracy_xgb})



lr = LinearRegression()
lr.fit(x_trans_train,y_train)

# check the accuracy
y_pred_train_lr = lr.predict(x_trans_train)
training_accuracy_lr = r2_score(y_train,y_pred_train_lr)

y_pred_test_lr = lr.predict(x_trans_test)
test_accuracy_lr = r2_score(y_test,y_pred_test_lr)
print({'training_accuracy_lr': training_accuracy_lr, 'test_accuracy_lr': test_accuracy_lr})

# from statsmodels.stats.outliers_influence import variance_inflation_factor
# vif_data = pd.DataFrame()
# vif_data["feature"] = x.columns
# vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
# print(vif_data)