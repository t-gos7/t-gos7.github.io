---
layout: post
title: "Tarit Goswami, Software Developer - House Price Prediction"
date: 2020-02-03
---

## House price prediction using Python
I have used *Linear regression* for predicting house prices. Linear regression is a linear modelling approach to find relationships between one or more independent variables, denoted by X and dependent vars. Y. 

### Steps 
#### 1. Load the data
We will use `Pandas` as `pd` to load the `csv` file. There are 6 attributes in the file - `LotArea`(total area), `OverallQual`(a rating from 1-10 range), `YearBuilt`(year of establishment), `FullBath`(number of full bathroom), `Bedroom`(number of bedrooms) and `SalePrice`(selling price). We are going to build a model to predict the value of `SalePrice`, given other independent attributes. There are 1460 datums. 

```
dataset = pd.read_csv("house_data.csv")
#print(dataset.shape)
#print(dataset.head(10))
print(dataset.describe())

dataset.hist()
#plot.show()

array=dataset.values
X = array[:,:5]
Y = array[:,5] 
```

#### 2. Preprocess the data
To drop the datums which contains null value and maybe outlier(add error to the prediction model), we need to choose the datums we are going to keep to build the model with. This step is very crucial in order to make a good model. 

#### 3. Split the dataset into training and testing data
We need to split the data into training and testing data. I have used `train_test_split` from `sklearn.model_selection`. It takes the `X`,`Y`, `test_size` and `random_state` as parameters.The `test_size` takes value between `0.0` and `1.0`. It's the proportion of data we need to use for testing our model, the rest will be used for training the model. 

```
validation_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)
```

There is another method to split the training and testing data. It's **K-fold cross validation**. We will pass a `k` value and the data will be devided into **k** equal subsets. Out of which, `(k-1)` parts will be used for training purpose and the one left will be used for testing purpose. Each time one part of **k** parts will be used for test purpose and rest will be used for training purpose and the best split will be chosed(according to the validation accuracy) for building the model. 
#### 4. Fit the data on simple regression model 
I have used `LinearRegression` for prdicting house price. 
```python 3
reg = linear_model.LinearRegression()
reg = reg.fit(X_train,Y_train)
predictions=reg.predict(X_test)       # Predicts outcome(here house price) for the test dataset 
```
#### 5. Predict accuracy 
```python 3
accuracy = reg.score(X_test, Y_test)  # Finds the residual sum squared error, i.e; (1-u/v) where u = sum((y_test - y_pred)^2)
print("Accuracy = ",accuracy)         # and v is sum(( y_test - mean(y_test) )^2) 
```
#### 6. Use in real scenario
Now it's time to see how the model can predict the house price when we give it the inputs.
