from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import mnist

poly = PolynomialFeatures(2)

# x = [[8,65],
#     [-44,-51],
#     [17,-33],
#     [-75,48],
#     [18,45],
#     [-92,-41],
#     [11,13],
#     [1,17],
#     [-80,-64],
#     [1,-31],
#     [-40,71],
#     [28,51],
#     [-60,-20],
#     [-71,-71]]
# r = [39,-84,-18,49,-29,16,73,-98,-57,-12,17,-43,1,-49]
# y = [x[i][0]*x[i][0]/10 + 5*x[i][1] + x[i][0]*x[i][1]/6 + r[i] for i in range(len(x))]
# x_expand = poly.fit_transform(x)
# X_train, X_test, Y_train, Y_test = train_test_split(x_expand, y, random_state = 0)

x_train, y_train, x_test, y_test = mnist.load()

print(len(x_train))
print(len(x_train[0]))
print(len(x_test))
print(len(x_test[0]))

num = 20000
inp = 300
test_num = 400

X_train = poly.fit_transform(x_train[0:num,0:inp])
print(len(X_train))
print(len(X_train[0]))
X_test = poly.fit_transform(x_test[0:test_num,0:inp])
Y_train_0 = list(map(lambda y: 1 if y == 0 else 0, y_train))
Y_test_0 = list(map(lambda y: 1 if y == 0 else 0, y_test))

linreg = Ridge(alpha=1.0).fit(X_train, Y_train_0[0:num])

print('(poly deg 2 + ridge) linear model coeff (w):\n{}'
     .format(linreg.coef_))
print('(poly deg 2 + ridge) linear model intercept (b): {:.3f}'
     .format(linreg.intercept_))
print('(poly deg 2 + ridge) R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, Y_train_0[0:num])))
print('(poly deg 2 + ridge) R-squared score (test): {:.3f}'
     .format(linreg.score(X_test, Y_test_0[0:test_num])))

# DONE
# Y_train how does label work

# TODO
# polyfit only individual terms, scale, get covariances, cut all terms with covariance

# Pre filter data: PCA, Decision Tree, Variance, Correlation with output
# Scale inputs
# Ridge regression lambda changes by number of terms
# Post filter data
