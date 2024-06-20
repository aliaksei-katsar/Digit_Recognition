# Digit recognition using Logistic Regression, Softmax and Gradient Descent
> This repository contains Python scripts for implementing softmax and logistic regression models to recognize handwritten digits from the MNIST dataset.
> 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Navigation

- [Overview](#overview)
- [Features](#Features)
- [Conclusion](#Conclusion)

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

- **Implementation**: Both softmax and logistic regression models were implemented in Python without relying on external libraries except for data handling and visualization.
- **Performance**: Achieved accuracy scores of 93% and 99% on the test set for softmax and logistic regression models, respectively.
- **Optimization**: Explored and compared different step size functions for gradient descent optimization.

## Results

