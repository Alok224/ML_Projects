import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'c:\Users\HP\Downloads\boxoffice.csv')
# print(df)
# check the null values
# null_values = df.isnull().sum()
# print(null_values)

# print(df.info())
# print(df.describe())

# Now, time to data cleaning
# I want to remove the dollar signs from the samples in domestic_revenue columns
df['domestic_revenue'] = df['domestic_revenue'].astype(str).str.replace('$',' ')
# removes the commas from the columns
columns = ['domestic_revenue','opening_theaters','release_days']
for i in columns:
    df[i] = df[i].astype(str).str.replace(',',' ')

    # Now, convert the all datapoints into float datatype
    df[i] = df[i].astype(float)
    df[i] = pd.to_numeric(df[i], errors = 'coerce')

# print(df.info())
# checking the outliers

plt.subplots(figsize = (10,6))
for i, col in enumerate(columns):
    plt.subplot(1,3,i+1)
    sns.displot(df[col])
plt.tight_layout()
# plt.show()

plt.subplots(figsize = (10,6))
for i, col in enumerate(columns):
    plt.subplot(1,3,i+1)
    sns.boxplot(df[col])
plt.tight_layout()
# plt.show()

from sklearn.feature_extraction.text import CountVectorizer
# intialize the countvectorizer
cvz = CountVectorizer()
cvz.fit(df['genres'])
# Now, convert into array format
transform_categories = cvz.transform(df['genres']).toarray()
# Now create the seperate list for the categories in geners'genres'
# first extract the all categories
extract_categories = cvz.get_feature_names_out()
for i, ex_cg in enumerate(extract_categories):
    df[ex_cg] = transform_categories[:,i]
# drop the column geners'genres'
df.drop('genres',axis = 1)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
feat = ['distributor','MPAA']
for i in feat:
    df[i] = le.fit_transform(df[i])

x = df.drop(['domestic_revenue','title'],axis=1)
y = df['domestic_revenue']

for column in x.columns:
    if x[column].dtype == 'object':
        x[column] = le.fit_transform(x[column])

from sklearn.model_selection import train_test_split, GridSearchCV
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)
# check the correlations between the features
plt.figure(figsize=(12,10))
correlation_matrix = x_train.corr()
strong_corr = correlation_matrix[(correlation_matrix > 0.7) | (correlation_matrix < 0.7)]
sns.heatmap(correlation_matrix,annot = True,cmap=plt.cm.CMRmap_r)
# plt.show()

# delete the correlation fetaures
# create the set for fetch the correlation features
corr_features = set()
for i in range(len(correlation_matrix)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i,j]>0.7):
            colname = correlation_matrix.columns[i]
            corr_features.add(colname)
print('Highly correlated features:', corr_features)
# Not having any correlation feautres
# print(x_train.shape)
# print(x_test.shape)

# Now, convert the datapoints in some features which have very high scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train_transformed_data = ss.fit_transform(x_train)
x_test_transformed_data = ss.transform(x_test)

# Now, let's build the model
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.01,
    reg_lambda=1,
    random_state=42)
# parameters = {
#                 'n_estimators' : [100,200,250,300],
#                 'learning_rate' : [0.5,1.0,0.01,0.3],
#              }
# grid = GridSearchCV(XGBRegressor(),parameters,cv = 5)
# grid.fit(x_train_transformed_data,y_train)
# print(grid.best_params_)
# print(grid.best_score_)
model.fit(x_train_transformed_data,y_train)

# Now, let's evaluate the model
y_pred = model.predict(x_test_transformed_data)

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_pred,y_test)
print(mse)
r2s = r2_score(y_pred,y_test)
print(r2s)