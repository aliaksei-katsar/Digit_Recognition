# Digit recognition using Logistic Regression, Softmax and Gradient Descent
> This repository contains Python scripts for implementing softmax and logistic regression models to recognize handwritten digits from the MNIST dataset.
> 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Navigation

- [Overview](#overview)
- [Features](#Features)
- [Conclusion](#Conclusion)
- [Results](#Results)

## Overview

### Goals

- To implement Softmax function to distinguisch between the numbers in the dataset.
- To implement gradient descent and Newton methods to minimize functions.
- To try different stepwise methods(Powel Wolfe, Armijo, constant) and understand their pros and cons.

### Dataset 

- MNIST dataset

### Model

- **Softmax Regression**: A multi-class logistic regression model using the softmax function.
- **Logistic Regression**: A binary logistic regression model for digit classification.
- **Gradient Descent**: An optimization algorithm used to minimize a function iteratively by adjusting its parameters.
- **Armijo method**: A line search method in optimization that ensures sufficient decrease of the objective function by adjusting the step size iteratively.
- **Powel Wolfe method**: A line search method in optimization that combines the desirable properties of the Powell method for finding a local minimum and the Wolfe conditions for ensuring sufficient decrease and curvature.

## Features

- **Implementation**: Both softmax and logistic regression models were implemented in Python without relying on external libraries except for numpy.
- **Performance**: Achieved accuracy scores of 92% and 99% on the test set for softmax and logistic regression models, respectively.
- **Optimization**: Explored and compared different step size functions for gradient descent optimization.

## Results

- **Logistic Regression with constant learning rate**:
  
![constant_lreg]
- **Softmax Regression with constant learning rate**:
  
![constant_softmax]
- **Logistic Regression with Armijo step size**:
  
![armijo_lreg]
- **Softmax Regression with Armijo step size**:
  
![armijo_softmax]
- **Logistic Regression with Powell-Wolfe step size**:
  
![powell_wolfe_lreg]
- **Softmax Regression with Powell-Wolfe step size**:
  
![powell_wolfe_softmax]

## Conclusion

This project focuses on implementing and optimizing machine learning models for digit recognition using softmax regression and logistic regression techniques. Implemented from scratch in Python, these models achieved significant accuracy scores of 92% and 99% on test datasets for softmax and logistic regression, respectively, showcasing their effectiveness in classifying handwritten digits from the MNIST dataset.

Optimization techniques such as gradient descent with the Armijo and Powell-Wolfe methods were explored to fine-tune model performance. This project provided valuable insights into implementing machine learning algorithms, handling data preprocessing, and optimizing model parameters to enhance accuracy and efficiency.

For logistic regression, the Armijo and Powell-Wolfe methods proved effective. However, for softmax regression, these methods were less effective as they required more time to converge and didn't show significant results compared to using a constant learning rate.



[constant_lreg]: results/constant_logistic_regression.png
[constant_softmax]: results/constant_softmax.png
[armijo_lreg]: results/armijo_logistic_regression.png
[armijo_softmax]: results/armijo_softmax.png
[powell_wolfe_lreg]: results/powell_wolfe_logistic_regression.png
[powell_wolfe_softmax]: results/powell_wolfe_softmax.png
