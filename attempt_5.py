from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import mnist

"""
Machine Learning Steps

1. Gather data
  a. x_train (train_inputs x features)
  b. y_train (train_inputs)
  c. x_test (test_inputs x features)
  d. y_test (test_inputs)
2. Scale the values
3. Determine if using feature by variance
4. For all remaining features, polyfit combine features to desired degree
5. Remove low variance
6. Compute linear (or Lasso) regression
7. Remove small terms
8. Re-compute regression
"""

degrees = 3
threshold1 = 0.188
threshold2 = 0.18

print("Gather Data")
x_train, y_train, x_test, y_test = mnist.load()

print("Scale inputs")
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)

print("Determine if using feature")
selector1 = VarianceThreshold(threshold = threshold1) # TODO: come up with a better way to set threshold
x_train_scaled_filterd = selector1.fit_transform(x_train_scaled)
print(len(x_train_scaled_filterd[0]))

print("Polyfit remaining features")
poly = PolynomialFeatures(degrees)
x_train_filtered_polyfit = poly.fit_transform(x_train_scaled_filterd)
print(len(x_train_filtered_polyfit[0]))

selector2 = VarianceThreshold(threshold = threshold2) # TODO: come up with a better way to set threshold
x_train_filtered_polyfit_filtered = selector2.fit_transform(x_train_filtered_polyfit)
print(len(x_train_filtered_polyfit_filtered[0]))

test_predictions = [(-1, -1)] * len(y_test)
for i in range(10):
    print(f"Compute linear (or Lasso) regression for {i}")
    y_train_0 = list(map(lambda y: 1 if y == i else 0, y_train))
    y_test_0 = list(map(lambda y: 1 if y == i else 0, y_test))
    linreg = LinearRegression().fit(x_train_filtered_polyfit_filtered, y_train_0)

    # print(linreg.coef_)
    # print(linreg.score(x_train_filtered_polyfit, y_train_0))

    print(f"Compute predictions inputs for {i}")
    x_test_scaled = scaler.fit_transform(x_test)
    x_test_scaled_filtered = selector1.transform(x_test_scaled)
    x_test_filtered_polyfit = poly.fit_transform(x_test_scaled_filtered)
    x_test_filtered_polyfit_filtered = selector2.transform(x_test_filtered_polyfit)
    predictions = linreg.predict(x_test_filtered_polyfit_filtered)
    for j in range(len(predictions)):
        if predictions[j] > test_predictions[j][0]:
            test_predictions[j] = (predictions[j], i)

total_correct = 0
total_wrong = 0
for i in range(len(test_predictions)):
    if y_test[i] == test_predictions[i][1]:
        total_correct += 1
    else:
        total_wrong += 1
        print(f"Actual: {y_test[i]}, Prediction: {test_predictions[i][1]}, Prob: {test_predictions[i][0]}")

print(f"Total Correct: {total_correct}")
print(f"Total Wrong: {total_wrong}")

"""
Settings: 
degrees = 3
threshold1 = 0.188
threshold2 = 0.18

Actual: 5, Prediction: 3, Prob: 0.4230268301210899
Actual: 3, Prediction: 8, Prob: 0.42247064247069255
Actual: 3, Prediction: 9, Prob: 0.4271689784966149
Actual: 2, Prediction: 6, Prob: 0.32930401667024606
Actual: 8, Prediction: 1, Prob: 0.5515845472209231
Actual: 5, Prediction: 3, Prob: 0.3997264967969284
Actual: 3, Prediction: 2, Prob: 0.3943949404220982
Actual: 6, Prediction: 7, Prob: 0.2670499204120399
Actual: 4, Prediction: 7, Prob: 0.42784456809372856
Actual: 2, Prediction: 6, Prob: 0.31958665189848057
Actual: 5, Prediction: 3, Prob: 0.2924113828163391
Actual: 3, Prediction: 8, Prob: 0.4724387991629152
Actual: 0, Prediction: 5, Prob: 0.2633455447327007
Total Correct: 7326
Total Wrong: 2674
"""

