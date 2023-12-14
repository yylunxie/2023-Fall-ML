import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

var = 0.07

noise = np.random.normal(0, np.sqrt(0.07), 30)


def g(t, noise):
    return np.square(np.sin(2*np.pi*t)) + noise


T = np.linspace(0, 1, 30)

S = np.zeros(30)
for i in range(30):
    S[i] = g(T[i], noise[i])

degrees = [2, 5, 10, 14, 18]
T = T.reshape(-1, 1)
plt.figure(figsize=(15, 10))
for i in range(5):
    polynomial_features = PolynomialFeatures(
        degree=degrees[i], include_bias=False)
    linear_regression = LinearRegression()
    pipeline = make_pipeline(polynomial_features, linear_regression)
    pipeline.fit(T, S)
    s_pred = pipeline.predict(T)

    plt.subplot(3, 2, i+1)
    plt.scatter(T, S, facecolors='none', edgecolors='#7B90D2')
    plt.plot(T, s_pred, c='#ED784A')
    plt.xlim((0, 1))
    plt.xlabel('t period', fontsize=16)
    plt.ylabel('signal data', fontsize=16)
    plt.title(
        f'Linear Regression Relation of Degree {degrees[i]} polynomial model', fontsize=16)

plt.tight_layout()
plt.savefig('5curves.png', dpi=300)
plt.close()
