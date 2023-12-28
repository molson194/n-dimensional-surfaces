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
5. Compute linear (or Lasso) regression
6. Remove small terms
7. Re-compute regression
"""

degrees = 2
threshold = 0.19

print("Gather Data")
x_train, y_train, x_test, y_test = mnist.load()

print("Scale inputs")
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)

print("Determine if using feature")
selector = VarianceThreshold(threshold = threshold) # TODO: come up with a better way to set threshold
x_train_scaled_filterd = selector.fit_transform(x_train_scaled)
print(len(x_train_scaled_filterd[0]))

print("Polyfit remaining features")
poly = PolynomialFeatures(degrees)
x_train_filtered_polyfit = poly.fit_transform(x_train_scaled_filterd)

test_predictions = [(-1, -1)] * len(y_test)
for i in range(10):
    print(f"Compute linear (or Lasso) regression for {i}")
    y_train_0 = list(map(lambda y: 1 if y == i else 0, y_train))
    y_test_0 = list(map(lambda y: 1 if y == i else 0, y_test))
    linreg = LinearRegression().fit(x_train_filtered_polyfit, y_train_0)

    # print(linreg.coef_)
    # print(linreg.score(x_train_filtered_polyfit, y_train_0))

    print(f"Compute predictions inputs for {i}")
    x_test_scaled = scaler.fit_transform(x_test)
    x_test_scaled_filtered = selector.transform(x_test_scaled)
    x_test_filtered_polyfit = poly.fit_transform(x_test_scaled_filtered)
    predictions = linreg.predict(x_test_filtered_polyfit)
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
Settings: Threshold 0.185, Degrees 2

Actual: 3, Prediction: 8, Prob: 0.6131973266601562
Actual: 7, Prediction: 9, Prob: 0.4571800231933594
Actual: 3, Prediction: 8, Prob: 0.40311431884765625
Actual: 8, Prediction: 2, Prob: 0.482100248336792
Actual: 3, Prediction: 8, Prob: 0.44600868225097656
Actual: 3, Prediction: 5, Prob: 0.4862518310546875
Actual: 2, Prediction: 9, Prob: 0.4483528137207031
Actual: 5, Prediction: 6, Prob: 0.395050048828125
Actual: 3, Prediction: 8, Prob: 0.3676624298095703
Actual: 2, Prediction: 5, Prob: 0.33168792724609375
Actual: 5, Prediction: 3, Prob: 0.4055328369140625
Total Correct: 9293
Total Wrong: 707
"""
