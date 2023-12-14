import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
nrow = 2

x = np.array([1, 3, 6, 9, 12, 15, 18, 21, 24, 30,
             36, 48, 60, 72, 84, 96, 108, 120])
y = np.array([7.571, 7.931, 7.977, 7.996, 8.126, 8.247, 8.298, 8.304, 8.311, 8.327, 8.369, 8.462,
              8.487, 8.492, 8.479, 8.510, 8.507, 8.404])
degrees = [1, 2, 3, 4, 5, 6]

x_p = np.linspace(1, 120, 1000)

x = x.reshape(-1, 1)
y_preds = []

r2 = []
plt.figure(figsize=(30, 10))
for i, degree in enumerate(degrees, 1):
    print(f"Degree {degree} polynomial: ")
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    pipeline = make_pipeline(polynomial_features, linear_regression)
    pipeline.fit(x, y)
    y_pred = pipeline.predict(x)
    y_pred_p = pipeline.predict(x_p.reshape(-1, 1))
    y_preds.append(y_pred_p)

    residual = y_pred - y
    corr_coefficient = np.corrcoef(y, y_pred)[0, 1]
    print(f"Correlation Coefficient: {corr_coefficient:.4f}")
    r2.append(corr_coefficient)
    # plt.subplot(nrow, len(degrees), i)
    # plt.hist(residual, bins=8)
    # plt.ylabel('frequency', fontsize=16)
    # plt.xlabel('residual', fontsize=16)
    # plt.title(f'Degree {degree}', fontsize=18)

    # plt.subplot(nrow, len(degrees), i+6)
    # stats.probplot(residual, dist="norm", plot=plt)
    # plt.title("Q-Q plot", fontsize=18)
    # plt.xlabel('Quantiles', fontsize=16)
    # plt.ylabel('Ordered Values', fontsize=16)

    # if degree == 4:
    #     residual = y - y_pred
    #     plt.figure(figsize=(7.5, 5.5))
    #     plt.subplot()
    #     plt.scatter(x, residual)
    #     plt.xlabel('maturity', fontsize=16)
    #     plt.ylabel('residual', fontsize=16)
    #     plt.title(
    #         "Residual plot for 4$^{th}$ order polynomial model", fontsize=18)

    #     plt.savefig('residual vs maturity', dpi=300)
    #     plt.close()

# plt.tight_layout()
# plt.savefig('histogram.png', dpi=300)
# plt.close()

# Figure 3
# plt.figure(figsize=(6, 6))
# plt.subplot()
# plt.scatter(x, y)
# plt.xlabel('maturity', fontsize=16)
# plt.ylabel('yields', fontsize=16)
# plt.title('yields vs maturity', fontsize=18)
# plt.tight_layout()
# plt.savefig('yields vs maturity', dpi=300)
# plt.close()

# Figure 4
# plt.figure(figsize=(6.5, 5.5))
# plt.subplot()
# plt.scatter(degrees, r2)
# plt.xlabel('polynomial order ${k}$', fontsize=16)
# plt.ylabel('${R}^{2}$', fontsize=16)
# plt.title('${R}^{2}$ vs the polynomial order ${k}$', fontsize=18)

# plt.savefig('𝑅2 vs the polynomial order 𝑘', dpi=300)
# plt.close()

plt.figure(figsize=(15, 10))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.scatter(x.reshape(-1), y)
    plt.plot(x_p, y_preds[i], c='#0F4C3A')
    plt.title(f'Degree {i+1} polynomial model', fontsize=18)
    plt.xlabel('maturity', fontsize=16)
    plt.ylabel('yields', fontsize=16)
    plt.ylim(min(y)-0.5, max(y)+0.5)

plt.tight_layout()
plt.savefig('poly.png', dpi=300)
