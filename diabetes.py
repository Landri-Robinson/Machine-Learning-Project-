''' Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line '''

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets


#how many sameples and How many features?
diabetes = datasets.load_diabetes()
print(diabetes.data.shape)  # (442, 10) -> 442 samples, 10 features

# What does feature s6 represent?
# s6 = blood sugar level (serum glucose measurement)
print(diabetes.feature_names)  # shows all feature names including 's6'


# use s6 (index 9) as the single feature for regression
s6 = diabetes.data[:, 9].reshape(-1, 1)
target = diabetes.target

model = LinearRegression()
model.fit(s6, target)

#print out the coefficient
print("Coefficient:", model.coef_)

#print out the intercept
print("Intercept:", model.intercept_)

# create a scatterplot with regression line
predicted_line = model.predict(s6)

plt.scatter(s6, target, color='steelblue', alpha=0.5, label='Data points')
plt.plot(s6, predicted_line, color='red', label='Regression line')
plt.xlabel('s6 (Blood Sugar Level)')
plt.ylabel('Disease Progression')
plt.title('Diabetes: s6 vs Disease Progression')
plt.legend()
plt.tight_layout()
plt.show()

