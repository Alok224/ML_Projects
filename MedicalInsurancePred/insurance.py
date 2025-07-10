import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\HP\Desktop\Machine learning\ml_practice\Medical_insurance.csv')
# print(df.shape)

# now, we need to data cleaning
# print(df.isnull().sum())
# print(df.info())

# convert the datatype float to int
df['bmi'] = df['bmi'].astype(int)
df['charges'] = df['charges'].astype(int)

# print(df.info())

# check the outliers
# sns.boxplot(df['age'])
# sns.boxplot(df['charges'])
# sns.boxplot(df['bmi'])

plt.show()

Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)
IQR = Q3 - Q1  # Compute IQR

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_no_outliers = df[(df['bmi'] >= lower_bound) & (df['bmi'] <= upper_bound)]

# sns.boxplot(df_no_outliers['bmi'])
# sns.boxplot(df_no_outliers['bmi'])
# plt.show()

QA1 = df['charges'].quantile(0.25)
QA3 = df['charges'].quantile(0.75)
IQR = QA3 - QA1
lower_bound1 = QA1 - 1.5 * IQR
upper_bound1 = QA1 + 1.5 * IQR



from feature_engine.outliers import ArbitraryOutlierCapper
capper = ArbitraryOutlierCapper(min_capping_dict = {'bmi': lower_bound,
                                                    'charges' : lower_bound1},                  
                                max_capping_dict={'bmi': upper_bound,
                                                  'charges': upper_bound1})
df[['bmi','charges']] = capper.fit_transform(df[['bmi','charges']])
# df[['charges']] = capper.fit_transform(df[['charges']])
# sns.boxplot(df['bmi'])
# plt.show()

sns.boxplot(df['charges'])
# plt.show()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['region'] = le.fit_transform(df['region'])
df['smoker'] = le.fit_transform(df['smoker'])

# print(df['sex'])

# Now, its time to create the model

# Splits the dataset into dependent and independent variables
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
# print(y)

# splits the data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

list1 = []
list2 = []
list3 = []
for i in range(40,50):
    x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=i)
    linear_model = LinearRegression()
    linear_model.fit(x_train,y_train)
    y_pred = linear_model.predict(x_test)
    # mse = mean_squared_error(y_pred,y_test)
    train_accuracy = linear_model.score(x_train,y_train)
    # r2s = r2_score(y_pred,y_test)
    test_accuracy = linear_model.score(x_test,y_test)
    list1.append(train_accuracy)
    list2.append(test_accuracy)
    # list3.append(r2s1)
    # create the dataframe for this
    df1 = pd.DataFrame({'train_accuracy' : list1, 'test_accuracy': list2})
# print(df1)

# Just take random state 47

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=47)
linear_model = LinearRegression()
linear_model.fit(x_train,y_train)
y_pred = linear_model.predict(x_test)
y_train_pred = linear_model.predict(x_train)
train_accuracy = r2_score(y_train, y_train_pred)
# test_accuracy = linear_model.score(x_test,y_test)
test_accuracy = r2_score(y_test,y_pred)
# print({'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy})
# {'train_accuracy': 0.70845292554744, 'test_accuracy': 0.7686242213865289}



# I want to apply the svr model
svr = SVR()
svr.fit(x_train,y_train)
# Now check the r2_score for the accuracy
# So, this is regression dataset

# training accuracy
y_train_pred_svr = svr.predict(x_train)
train_accuracy = r2_score(y_train,y_train_pred_svr)

y_test_pred_svr = svr.predict(x_test)
test_accuracy = r2_score(y_test,y_test_pred_svr)

# print({'training accuracy' : train_accuracy, 'test accuracy' : test_accuracy})
# So, this is not a perfect model for this dataset

# Now, try random forest reggressor

rdr = RandomForestRegressor(random_state=47)
# Apply gridsearchcv
parameters = {'n_estimators':[100,150,200,250,300],
              'max_depth': [2,3,4,5,6],
              }
grid_search = GridSearchCV(rdr,parameters,cv = 5,scoring='neg_mean_squared_error')
grid_search.fit(x_train,y_train)
# print(grid_search.best_params_)
# {'max_depth': 6, 'n_estimators': 100}

rdr = RandomForestRegressor(n_estimators=100, max_depth=6,random_state=47)
rdr.fit(x_train,y_train)
# check the accuracy(training and testing)
y_train_pred_rdr = rdr.predict(x_train)
training_accuracy_rdr = r2_score(y_train,y_train_pred_rdr)

y_test_pred_rdr = rdr.predict(x_test)
test_accuracy_rdr = r2_score(y_test,y_test_pred_rdr)

# print({'training_accuracy_rdr': training_accuracy_rdr, 'test_accuracy_rdr': test_accuracy_rdr})
# {'training_accuracy_rdr': 0.8085152506318515, 'test_accuracy_rdr': 0.824836697579327}

# Now, try to implement XGBRegressor
xgbr = XGBRegressor()
# Apply the gridseacrhcv
parameters = {
    'n_estimators': [50, 100, 150],  
    'learning_rate': [0.01, 0.1, 0.2, 0.3],  
    'max_depth': [2, 3, 4]
}
grid_search_xgbr = GridSearchCV(xgbr,parameters,cv = 5, scoring='neg_mean_squared_error')
grid_search_xgbr.fit(x_train,y_train)
# print(grid_search_xgbr.best_params_)
# {'learning_rate': 0.5, 'max_depth': 6, 'n_estimators': 100}
xgbr = XGBRegressor(learning_rate= 0.3, max_depth= 4, n_estimators= 50,random_state = 47)
xgbr.fit(x_train,y_train)

# check the accuracy(training and testing)
y_train_pred_xgbr = xgbr.predict(x_train)
training_accuracy_xgbr = r2_score(y_train,y_train_pred_xgbr)

y_test_pred_xgbr = xgbr.predict(x_test)
test_accuracy_xgbr = r2_score(y_test,y_test_pred_xgbr)
# print({'training_accuracy': training_accuracy_xgbr, 'test_accuracy': test_accuracy_xgbr})
# {'training_accuracy': 0.8576561808586121, 'test_accuracy': 0.8274837136268616}


from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(random_state=47)

parameters = {
                'n_estimators': [50,100,150,200,250],
                'learning_rate': [0.1,0.5,0.01,1.0],
                'max_depth': [2,3,4]
             }

grid_search_gbr = GridSearchCV(gbr,parameters,cv = 5, scoring='r2')
grid_search_gbr.fit(x_train,y_train)
# print(grid_search_gbr.best_params_)
# {'learning_rate': 0.3, 'max_depth': 4, 'n_estimators': 150}

gbr = GradientBoostingRegressor(learning_rate = 0.2, max_depth= 4, n_estimators= 21)

gbr.fit(x_train,y_train)
# to check the accuracy
y_pred_train_gbr = gbr.predict(x_train)
training_accuracy_gbr = r2_score(y_train,y_pred_train_gbr)

# to check the test accuracy
y_pred_test_gbr = gbr.predict(x_test)
test_accuracy_gbr = r2_score(y_test,y_pred_test_gbr)

# print({'training_accuracy': training_accuracy_gbr,'test_accuracy_gbr': test_accuracy_gbr})

new_data=pd.DataFrame({'age':19,'sex':'male','bmi':27.9,'children':0,'smoker':'yes','region':'northeast'},index=[0])
new_data['smoker']=new_data['smoker'].map({'yes':1,'no':0})
new_data['sex'] = new_data['sex'].map({'male':0,'female':1})
new_data['region'] = le.fit_transform(new_data['region'])

# new_data=new_data.drop(new_data[['sex','region']],axis=1)


pred = xgbr.predict(new_data)

print(pred)