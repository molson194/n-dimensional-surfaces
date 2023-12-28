# Fitting N-Dimensional Surfaces

TL;DR: I attempted combinations of 3 different ideas (Multi-Decision Trees, Tailor-Series Regression, and Neareast-Neighbor Optimization) to fit a surface to N-dimensional data (on the MNIST handwritten number dataset). These methods proved difficult computationally and non-optimal. My new idea: neural networks can create the optimal surface.

## Multi-Decision Trees

Today: Decision trees split on **one** variable at a **single** point. My Take: Often times, one variable isn't valuable in splitting data, let alone bifurcating that data.

Core Ideas:
1. Split data on multiple variables (ex: 0 < x1 < 2 && 1 < x2 < 3). Use correlation matrix to decide variables to combine. Correlated variables are likely to have overlapping predictions.
2. Instead of splitting left and right, split data on any continuous segments (i.e. inside range and outside range). Error calculations are the same - combine error about average for inside and outside of range. Work down the leaf with the highest error.

Additional Ideas
* Handling classification data: Use probabilities for the 'average' values (ex: 80% A, 15% B, 5% C - error can be computed from that)
* Missing Data: Split tree on HasData (and possibly combine with other variables), then use the data on the HasData side

Open questions:
* Decision trees create 'flat' segments of areas with the same prediction. Instead, should the prediction be interpolated from the boundaries?
* Should 'centered/mirrored data' be interpreted differently? Probably - correlations between inputs might be different. If magnitude is all that matters, could split on x^2.
* Temporal data using feedback loops?
* Reinforcement learning?

Links
* [Compare Machine Learning Algorithms](https://mljar.com/machine-learning/compare-ml-algorithms/)

## Taylor-Series Regression

Let's assume that we have 2 input variables (x1 and x2) and one output variable y. We can define an equation:

y = a * x1 + b * x2 + c * x1 * x2 + d * x1 ^ 2 + e * x2 ^2 ...

We can find a,b,c,d,... by minimizng the error between the observed points. However, this approach was not successful because:
  * Number of terms gets large too quickly
  * High computation time
  * Cannot handle missing data

## Nearest-Neighbor Optimization

Steps:
* For each point, create a 'ghost' point that is computed as a interpolation from the N nearest neighbors.
* Minimize the total error of each 'ghost' point, by computing the square of the difference between the 'ghost' point value and the actual value at each point.

Problems:
* Only looks at N neareast neighbors.
* Computationally intense
