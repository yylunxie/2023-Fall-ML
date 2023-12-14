import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import scipy.stats as stats

nrow = 4

X = np.loadtxt('02HW1_Xtrain')
y = np.loadtxt('02HW1_Ytrain')
X = X.reshape(-1, 1)

xp = np.linspace(min(X), max(X), 1000)

degrees = [1, 2, 3]
y_preds = []

plt.figure(figsize=(20, 8*nrow))

for i, degree in enumerate(degrees, 1):
    # print(f"Degree {degree} polynomial: ")
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    pipeline = make_pipeline(polynomial_features, linear_regression)
    pipeline.fit(X, y)
    y_pred = pipeline.predict(xp)
    y_preds.append(y_pred)
    y_pred_sample = pipeline.predict(X)
    training_error = mean_squared_error(y, y_pred_sample)
    residuals = y - y_pred_sample
    # print(f"Degree {degree} polynomial has training error {training_error:.2f}")

    # Row 1 -> residual
    plt.subplot(nrow, len(degrees), i)
    plt.ylim(-max(y), max(y))
    plt.axhline(y=0, color='r', linestyle='--')

    plt.xlabel('X', fontsize=16)
    plt.ylabel('Residuals', fontsize=16)
    plt.scatter(X, residuals)
    plt.title("Degree %d" % degree, fontsize=20)

    # Row 2 -> Normal distribution
    plt.subplot(nrow, len(degrees), i+len(degrees)*(1))
    plt.hist(residuals, bins=8)
    plt.xlabel('Residuals', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)

    # Row 3 -> QQ plot
    plt.subplot(nrow, len(degrees), i+len(degrees)*(2))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(" ", fontsize=18)
    plt.xlabel('Quantiles', fontsize=16)
    plt.ylabel('Ordered Values', fontsize=16)

    # Row 4 -> Independence
    plt.subplot(nrow, len(degrees), i+len(degrees)*(3))
    # plt.acorr(residuals)
    # plt.xlabel('Lags')
    # plt.ylabel('Correlation')
    plt.scatter(y_pred_sample, residuals)
    plt.xlabel("Fitted value", fontsize=16)
    plt.ylabel("Residuals", fontsize=16)
    plt.ylim(-6, 6)
    plt.axhline(y=0, color='r', linestyle='--')

    # Row 5 -> polynomial
    # plt.subplot(nrow, len(degrees), i+len(degrees)*(4))
    # plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    # plt.ylim(min(y)-1, max(y)+1)
    # plt.axhline(y=0, color='r', linestyle='--')
    # plt.plot(xp, y_pred, label="Polynomial of degree %d" % degree)
    # plt.xlabel("X", fontsize=16)
    # plt.ylabel("y", fontsize=16)
    # plt.legend(loc="best")

plt.tight_layout()
plt.savefig('HW02_1_a.png', dpi=300)
plt.close()

plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title("Degree %d" % (i+1), fontsize=20)
    plt.scatter(X, y, edgecolor='b', s=20)
    plt.ylim(min(y)-1, max(y)+1)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.plot(xp, y_preds[i])
    plt.xlabel("X", fontsize=16)
    plt.ylabel("Y", fontsize=16)

plt.tight_layout()
plt.savefig('HW02_1_b.png', dpi=300)
