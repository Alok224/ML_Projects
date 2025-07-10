import matplotlib.pyplot as plt
from PIL import Image
x = [4,5,6,7,8,9]
y = [5,1,3,4,6,7]
# Plotting the graph in x and y
# Linear graph
plt.plot(x,y,marker = '.')
plt.plot(x,marker= 'x')
plt.plot(y,marker = '>')
plt.show()

# marker is the corner points
# langchan
# Bar graph
x = [2,3,4,5]
y = [6,7,8,9]
c = ['r','b','g','y']
plt.bar(x,y,color = c)
plt.show()

import numpy as np
x = np.array(['A','B','C','D'])
y = np.array([3,4,5,6])
c = ['b','g','y','b']
# horizontal representations
plt.barh(x,y,color = c)
plt.show()

# To adjust the width of the graph
x = np.array(['A','B','C','D'])
y = np.array([2,3,4,5])
plt.bar(x,y,width=0.4)
plt.show()


x = [1,2,3,4,5,6]
y = [4,5,3,2,5,6]
# To give the title
plt.title('Scatter plot')
# To add the label in x-axis and y-axis
plt.xlabel('month')
plt.ylabel('year')
c = ['red','black','yellow','gray','pink','blue']
# To change the sizes of the points
sizes = [20,45,65,43,21,34]
plt.scatter(x,y,color = c,s = sizes)
plt.show()


fname = r'ml_practice/chamelion.jpeg'
image = Image.open(fname).convert('L')
# Mapping image into gray scale
plt.imshow(image,cmap='gray')
plt.imshow(image,cmap = 'Greens')
# this code will give you the actual image
image = Image.open(fname).convert('RGB')
plt.title('chamelion image')
plt.xlabel('length')
plt.ylabel('breadth')
# to create the grid lines
plt.grid()
plt.imshow(image)
plt.show()

# pie chart
x= [12,3,45,6]
y = ['English','Hindi','Maths','Science']
c = ['red','blue','green','yellow']
# To show the portions of the subjects in pie chart we can use labels = y
plt.pie(x,labels=y,colors=c)
# to show the percentage?Innformation of the subjects
plt.legend()
# To show the graph or output
plt.show()

import seaborn as sns
var = [1,2,3,4,5,6,7,8,9,10]
var1 = [2,3,4,5,6,7,8,9,10,11]
plt.plot(var,var1,color = 'g')
plt.show()
sns.lineplot(x = var,y = var1)
plt.show()
import pandas as pd
var = [1,2,3,4,5,6,7,8]
var1 = [2,3,4,5,6,7,8,9]
df = pd.DataFrame({"var":var,"var1":var1})
sns.lineplot(x = var,y = var1,data = df)
plt.show()
print(df)

df1 = sns.load_dataset("penguins").head(34)
# for cateogrical data we can use builtin parameter hue
sns.lineplot(x = 'bill_length_mm',y = "flipper_length_mm",data = df1,hue='sex')
sns.lineplot(x = 'bill_length_mm',y = "flipper_length_mm",data = df1,hue='sex',style = 'sex',palette='hot',markers=['o','<'])
plt.grid()
plt.title("This is the line graph of penguins")
plt.show()
print(df1["island"])
print((df1))
# categorize according male and female with the help of parameter hue
sns.barplot(x = 'island',y = 'bill_length_mm',data = df1,hue = 'sex')
# if I want to change the order of the bar graphs
sns.barplot(x = 'island',y = 'bill_length_mm',data=df1,hue='sex',order=['Biscoe','Dream','Torgersen'])
plt.show()

# If we want to change the order of male and female in the bar graph
# we use the saturation parameter to fade the bar graph
sns.barplot(x = 'island',y = 'bill_length_mm',data = df1,hue = 'sex',order = ['Biscoe','Dream','Torgersen'],hue_order = ['Female','Male'],palette='Accent',saturation = 0.5)
plt.show()

# histogram in the seaborn library
# By using the bins we can give the order of the garph
# kde = kernel density which represents the density of the graph
sns.displot(data = df1, x = 'flipper_length_mm',bins = [170,180,190,200,210,220,230],kde = True)
plt.show()

# boston_dataset is basically a dataset of boston house(which is a classic dataset for regression analysis).
from sklearn.datasets import fetch_california_housing
df = fetch_california_housing()
print(type(df))
print(df)
print(pd.DataFrame(df.data))
print(pd.DataFrame(df))
dataset = pd.DataFrame(df.data)
dataset.columns = df.feature_names
# I want to add the target column in the dataset(target column means that the column which I want to predict)
dataset['Price'] = df.target
print(dataset.head())

# To divide the dataset into dependent and independent variables
# this is my independent features
x = dataset.drop(columns=['Price'])
# this is my dependent features
y = dataset['Price']
print(x.head())
print(y.head())



# Linear regression Algorithm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
lin_reg = LinearRegression()
mse = cross_val_score(lin_reg,x,y,scoring = 'neg_mean_squared_error',cv = 5)
# print(mse)
mse_mean = np.mean(mse)
print(mse_mean) 



# Ridge regression Algorithm
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge = Ridge()
parameters = {'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20]}
ridge_regressor = GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv = 5)
ridge_regressor.fit(x,y)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state = 42)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
lin_reg = LinearRegression()
mse = cross_val_score(lin_reg,x_train,y_train,scoring = 'neg_mean_squared_error',cv = 5)
print(mse)
mean_mse = np.mean(mse)
print(mean_mse)




# logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
df = load_breast_cancer()
dataset = pd.DataFrame(df.data,columns = df.feature_names)

print(dataset.head())

# I want to add the dependent feature
dataset['target'] = df.target
print(dataset.head(20))
# To check that the dataset is balanced or imbalanced
print(dataset['target'].value_counts())
# just divide the dependent amd independent features
x = dataset.drop(columns=['target'])
y = dataset['target']
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state=42)
params = [{'C': [1, 5, 10]}, {'max_iter': [100, 150]}]
model1 = LogisticRegression(C = 100,max_iter=200)
model = GridSearchCV(model1,param_grid = params,scoring = 'f1',cv = 5)
model.fit(x_train,y_train)
print(model.best_params_)
print(model.best_score_)
y_pred = model.predict(x_test)
print(y_pred)



# Confusion matrix
from sklearn.metrics import confusion_matrix ,classification_report,accuracy_score
c = confusion_matrix(y_pred,y_test)
print(c)
# If I want accuracy_score 
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

print(classification_report(y_test,y_pred))



from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge_regression = Ridge()
parameter = {'alpha':[1,2,6,7,9,45,67,89,76,42]}
ridgecv = GridSearchCV(ridge_regression,parameter,scoring='neg_mean_squared_error',cv = 5)
c = ridgecv.fit(x_train,y_train)
print(c)
print(ridgecv.best_params_)
print(ridgecv.best_score_)
ridge_pred = ridgecv.predict(x_test)

import seaborn as sns
sns.displot(ridge_pred-y_test,kind='kde')
plt.show()


df = sns.load_dataset('iris')
print(df.head)
print(df['species'].unique())
['setosa' 'versicolor' 'virginica']

print(df.isnull().sum())
# If I want to remove the soecies of setosa then
df1 = (df[(df['species']!='setosa')])
print(df1.head())

df2 = df1['species'].map({'versicolor':0,'virginica':1})
print(df2.head())

# split the dataset into dependent and independent features
x = df1.drop(columns=['species'])
y = df2['species']
print(x.tail())
y = df2
print(y.tail())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

classifier = LogisticRegression()
parameter = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60, 56],
    'max_iter': [100, 150, 200, 250, 300]
}

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
classifier_regression = GridSearchCV(classifier, param_grid=parameter, scoring='accuracy', cv=5)
classifier_regression.fit(x_train, y_train)

print(classifier_regression)
print(classifier_regression.best_params_)
print(classifier_regression.best_score_)

# predictions
y_pred = (classifier_regression.predict(x_test))

# accuracy score

from sklearn.metrics import accuracy_score, classification_report
score = accuracy_score(y_pred,y_test)
print(score)
print(classification_report(y_test,y_pred))

sns.pairplot(df1,hue='species')
plt.show()



# naive bayes algorithm

df = pd.read_csv(r'C:\Users\HP\Downloads\Weather Dataset\Weather Dataset\weather.csv')
print(df.head())

x = df.drop(columns = ['Play'])
y = df['Play']
print(x.head())
print(y.head())


# to calculate the prior probability

prior_probability = df['RainTomorrow'].value_counts(normalize=True)
print(prior_probability)

train_size = x.shape[0]
class_priors = {}
for outcome in np.unique(y):
    outcome_count = sum(y == outcome)
    class_priors[outcome] = outcome_count / train_size
print(class_priors)

# computing likelihoods

features = list(x.columns)
likelihoods = {}
for outcome in np.unique(y):
    outcome_count = sum(y == outcome)
    for feature in features: 
        features_unique_values = np.unique(x[feature])
        for feature_value in features_unique_values:
            count = 0
            for i in range(len(x)):
                if(x[feature][i]) == feature_value and y[i] == outcome:
                    count = count + 1
            likelihoods[(feature,feature_value,outcome)] = (count + 1)/(outcome_count + (len(features)))
print(likelihoods)

# to find the count of the uniques values in target values

target_values = len(np.unique(y))
print(target_values)

# this is my test data

test_data = pd.read_csv(r'C:\Users\HP\Downloads\Weather Dataset\Weather Dataset\weather_test.csv')
print(test_data)

x_test = test_data.drop(columns = ['Play'])
y_test = test_data['Play']

prob = np.ones((target_values,len(x_test)),dtype=np.float64)
print(prob)

for outcome in(np.unique(y)):
    outcome_count = sum(y == outcome)
    for feature in features:
        for i in range(len(x_test)):
            if  (feature,x_test[feature][i],outcome) in likelihoods.keys():
                prob[outcome][i]=prob[outcome][i]*likelihoods[(feature,x_test[feature][i],outcome)]
            else:
                prob[outcome][i]=prob[outcome][i]*(1/(outcome_count+len(features)))
     



# Support vector machines algorithm


from sklearn.datasets import load_iris
df = load_iris()
dataset = pd.DataFrame(df.data,columns = df.feature_names)
print(dataset)

print(df.target_names)

dataset['flower_type'] = df.target
print(df.target_names)

dataset['flower_type'] = dataset['flower_type'].map({i: name for i, name in enumerate(df.target_names)})
print(dataset)

dataset1 = dataset[:50]
dataset2 = dataset[50:100]
dataset3 = dataset[100:]

import matplotlib.pyplot as plt
# %matplotlib inline

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

plt.scatter(dataset1['sepal length (cm)'], dataset1['sepal width (cm)'], color = "green",marker= '+')
plt.scatter(dataset2['sepal length (cm)'],dataset2['sepal width (cm)'], color = 'red',marker = '.')
plt.show()


# If I want to plot petal length and petal width

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')

plt.scatter(dataset1['petal length (cm)'], dataset1['petal width (cm)'], color = 'green',marker='+')
plt.scatter(dataset2['petal length (cm)'],dataset2['petal width (cm)'],color = 'red',marker = '.')
plt.show()


from sklearn.model_selection import train_test_split
x = dataset.drop(columns='flower_type')
y = dataset['flower_type']
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(len(x_train))
print(len(x_test))

from sklearn.svm import SVC
SVM_model = SVC()
SVM_model.fit(x_train, y_train)
SVM_model.score(x_test, y_test)

train_accuracy = SVM_model.score(x_train, y_train)
print(train_accuracy)
print(x_train)
print(y_train)
print(x_test)
print(y_test)
test_accuracy = SVM_model.score(x_test, y_test)
print(test_accuracy)

print(SVM_model.predict([[2.1,3.4,6.7,2.2]]))




import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
iris = load_iris()
dataset = pd.DataFrame(data = iris.data, columns= iris.feature_names)
dataset['flower_type'] = iris.target
dataset['flower_type'] = dataset['flower_type'].map({i: name for i, name in enumerate(iris.target_names)})
print(dataset)
print(sns.load_dataset('iris'))

x = dataset.drop(columns = 'flower_type')
y = dataset['flower_type']

# Now the data splits
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)
print(x_train)
print(x_test)
print(y_train)
print(y_test)


# Decision Tree Classifier Algorithm

from sklearn.tree import DecisionTreeClassifier
treemodel = DecisionTreeClassifier(max_depth=2)
treemodel.fit(x_train,y_train)

from sklearn import tree
plt.figure(figsize=(15,10))
print(tree.plot_tree(treemodel,filled = True))
print(dataset.head())
plt.show()


y_pred = treemodel.fit(x_test,y_test)
from sklearn import tree
plt.figure(figsize=(15,10))
print(tree.plot_tree(y_pred,filled = True))
print(dataset.head())
plt.show()

# to predict the values of test data 

y_pred = treemodel.predict(x_test)
print(y_pred)

# to check the accuracy of my model
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_pred,y_test)
print(accuracy)
print(classification_report(y_pred,y_test))

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
treemodel = DecisionTreeClassifier()
treemodel.fit(x_train,y_train)

parameters = {'criterion' : ['gini', 'entropy', 'log_loss'],
              'splitter' : ['best', 'random'],
              'max_depth': [1,2,3,4,5,6,7,8],
              'max_features': ['auto','sqrt','log2'],
              'ccp_alpha': [0.0,0.1,0.2,0.3,0.4,0.5]
              }
cv = GridSearchCV(treemodel,param_grid=parameters,scoring='accuracy',cv = 5)
cv.fit(x_train,y_train)
print(cv.best_params_)
y_pred = cv.predict(x_test)
print(y_pred)

# I want to check the accuracy
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_pred,y_test)
print(accuracy)

print(classification_report(y_pred,y_test))

import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sns
diabetes = load_diabetes()
dataset = pd.DataFrame(data = diabetes.data, columns= diabetes.feature_names)
dataset['diabetes_analysis'] = diabetes.target
dataset['flower_type'] = dataset['flower_type'].map({i: name for i, name in enumerate(diabetes.target_names)})
print(dataset)
print(sns.load_dataset('diabetes'))

x = dataset.drop(columns = 'diabetes_analysis')
y = dataset['diabetes_analysis']
print(y)

# Now the data splits
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
treemodel = DecisionTreeRegressor()
treemodel.fit(x_train,y_train)

parameters = { 
             'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
             'splitter': ['best', 'random'],
              'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],
             'max_features': ['sqrt', 'log2'],
             'ccp_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
           }
cv = GridSearchCV(treemodel,param_grid=parameters,scoring='neg_mean_squared_error',cv = 5)
cv.fit(x_train,y_train)
print(cv.best_params_)
y_pred_test = cv.predict(x_test)
y_pred_train = cv.predict(x_train)
print(y_pred_test)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
print(mse)
r2s = r2_score(y_test, y_pred)
print(r2s)
print(classification_report(y_pred,y_test))

mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)


print(f"Test Mean Squared Error: {mse_test}")
print(f"Test R^2 Score: {r2_test}")
print(f"Train Mean Squared Error: {mse_train}")
print(f"Train R^2 Score: {r2_train}")


if mse_train < mse_test and r2_train > r2_test:
    print("The model is likely overfitting.")
else:
    print("The model is not overfitting.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
df = load_iris()
dataset = pd.DataFrame(data = df.data, columns=df.feature_names)
print(dataset)
print(sns.load_dataset('iris'))

dataset['species'] = df.target
print(dataset)

# divide into dependent and independent features
x = dataset.drop(columns='species')
y = dataset['species']
print(y)


# Random Forest Classifier Algorithm
# Now split the data
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)
rdc = RandomForestClassifier()
parameters = {
    'n_estimators': [100, 200, 300, 400, 500],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': list(range(2, 15)),
    'min_samples_leaf': list(range(1, 10)),
    'bootstrap': [True, False],
    'oob_score': [True, False],
    'n_jobs': [-1],
    'random_state': [42],
    'max_features': ['sqrt', 'log2']  
}
grid_search = GridSearchCV(rdc, parameters, cv=5)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
y_pred = grid_search.predict(x_test)
predicted_trained_data = grid_search.predict(x_train)
from sklearn.metrics import accuracy_score,classification_report
accuracy = accuracy_score(y_pred,y_test)
print(accuracy)
print(classification_report(y_pred,y_test))

plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.scatter(y_pred,y_test,color = 'r')
plt.legend()
plt.scatter(y_test,color = 'b')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='r', label='Test Data')
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.legend()
plt.show()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

df = load_breast_cancer()
# in dataframe
dataset = pd.DataFrame(df.data, columns = df.feature_names)
print(dataset)
# Try to map the target output to the rows
dataset['output'] = df.target
print(dataset['output'])
print(df.target)
dataset['output_name'] = dataset['output'].map({i: name for i, name in enumerate(df.target_names)})
print(dataset['output_name'])
print(dataset)

# Now, divide the dataset into dependent and independent features

x = dataset.drop(columns = ['output','output_name'])
y = dataset['output_name']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

# Correctly split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Adaboost classifier Algorithm
# Initialize the AdaBoostClassifier
adaboost = AdaBoostClassifier()
lgc = LogisticRegression()
# Define the parameter grid
parameters = {
    'n_estimators': [50, 100, 150, 200],  # Number of weak learners
    'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # Learning rate for weight updates
    'algorithm': ['SAMME', 'SAMME.R']  # Include both options for comparison
}

# Perform GridSearchCV
cv = GridSearchCV(adaboost, param_grid=parameters, scoring='accuracy', cv=5)
cv.fit(x_train, y_train)

# Print the best parameters
print(cv.best_params_)
y_pred = cv.predict(x_test)

# Now check the accuracy 
from sklearn.metrics import accuracy_score,classification_report
accuracy = accuracy_score(y_pred,y_test)
print(accuracy)
print(classification_report(y_pred,y_test))


# Gradient boost regression algorithm


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Load the dataset
df = fetch_california_housing()
dataset = pd.DataFrame(data=df.data, columns=df.feature_names)
dataset['targeted_data'] = df.target

# Divide the dataset into dependent and independent features
x = dataset.drop(columns='targeted_data')
y = dataset['targeted_data']

# Correctly split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Initialize the GradientBoostingRegressor
gbr = GradientBoostingRegressor()

# Define the parameter grid
parameters = {
    'learning_rate': [0.15,0.1,0.01,0.05],
    'n_estimators': [100, 150],
}

# Perform GridSearchCV
grid = GridSearchCV(gbr, parameters, scoring='neg_mean_squared_error', cv=5)
grid.fit(x_train, y_train)

# Predict the values of the test data
# y_pred = grid.predict(x_test)
y_pred_train = grid.predict(x_train)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

y_pred_test = grid.predict(x_test)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(grid.best_params_)

print("Training Mean Squared Error:", mse_train)
print("Test Mean Squared Error:", mse_test)
print("Training R2 Score:", r2_train)
print("Test R2 Score:", r2_test)

# Evaluate the model

if mse_train < mse_test and r2_train > r2_test:
    print("The model is likely overfitting.")
else:
    print("The model is not overfitting.")
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print("Mean Squared Error:", mse)
# print("R2 Score:", r2)