# Price Prediction Using MLP

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Read Dataset](#read-dataset)
5. [Data Preprocessing](#data-preprocessing)
    1. [Splitting the Data](#Splitting-the-Data)
    2. [Scaling the Features](#Scaling-the-Features)
6. [Model Building](#model-building)
    1. [Creating the Model](#Creating-the-Model)
    2. [Compiling and Training the Model](#Compiling-and-Training-the-Model)
7. [Model Evaluation and Visualization](#model-evaluation-and-visualization)
    1. [Evaluating the Model](#Evaluating-the-Model)
    2. [Visualizing Training Loss](#Visualizing-Training-Loss)
    3. [Plotting True vs. Predicted Values](#Plotting-True-vs.-Predicted-Values)
9. [Conclusion](#conclusion)

## Introduction
This project aims to predict prices using a neural network model. The dataset used is `fake_reg.csv`. The process includes reading the dataset, preprocessing the data, building a neural network model, and evaluating its performance.

## Project Structure
- `Price Predication.ipynb`: Jupyter Notebook containing the entire workflow.
- `fake_reg.csv`: Dataset used for training and testing the model.

## Read Dataset
1. **Loading Libraries**:
   ```python
   import pandas as pd 
   import numpy as np 
   import matplotlib.pyplot as plt 
   from sklearn.preprocessing import MinMaxScaler, StandardScaler
   from sklearn.model_selection import train_test_split 
   from keras.models import Sequential 
   from keras.layers import *
   from keras.callbacks import EarlyStopping
   import seaborn as sns
   ```
2. **Reading the Data:**
   ```python

    df = pd.read_csv('fake_reg.csv')
    df.head()
    df.info()
    df.describe()
 
## Data Preprocessing
1. **Splitting the Data:**

   ```python
    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
2. **Scaling the Features:**

    ```python

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
## Model Building
 1.**Creating the Model:**

    ```python
    model = Sequential()
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
2. **Compiling and Training the Model:**

   ```python
    es = EarlyStopping(patience=20, monitor='loss')
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train_scaled, y_train, epochs=300, callbacks=[es])
## Model Evaluation and Visualization
1. **Evaluating the Model:**
    ```python
    model.evaluate(x_test_scaled, y_test)
    model.evaluate(x_train_scaled, y_train)
2. **Visualizing Training Loss:**

   ```python
    loss_df = pd.DataFrame(model.history.history)
    loss_df.plot()
    plt.show()
 3. **Plotting True vs. Predicted Values:**
    ```python
      y_pred = model.predict(x_test_scaled)
      y_true = pd.DataFrame(y_test.values, columns=['True Value'])
      y_predicate = pd.DataFrame(y_pred, columns=['Predicate_values'])
      dff = pd.concat([y_true, y_predicate], axis=1)
      plt.plot(dff['True Value'], color='red', label='True_values')
      plt.plot(dff['Predicate_values'], color='blue', label='Predicated Values')
      plt.legend()
      plt.xlabel('Time')
      plt.ylabel('Price')
      plt.show()
      sns.scatterplot(x=dff['True Value'], y=dff['Predicate_values'])
      plt.show()
## Conclusion
This project successfully demonstrates how to build and evaluate a neural network model for price prediction. The notebook covers data loading, preprocessing, model building, and evaluation, providing a clear and concise workflow for similar predictive modeling tasks.
