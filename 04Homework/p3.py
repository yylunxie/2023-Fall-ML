import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler


def PCA_k(X, k=None):
    pca = PCA(k)
    return pca.inverse_transform(pca.fit_transform(X))


# Load the .mat file
data = scipy.io.loadmat('04HW2_noisy.mat')

X = data['X'].T
id = [10, 121, 225, 318, 426]


# f, ax = plt.subplots(2, 5, figsize=(10, 5))
# for i in range(5):
#     ax[0, i].imshow(X[id[i], :].reshape(28, 20), cmap='gray')
#     ax[0, i].set_title('Original')
#     ax[0, i].axis('off')
#     ax[1, i].imshow(PCA_k(X, 10)[id[i], :].reshape(28, 20), cmap='gray')
#     ax[1, i].set_title('k=10')

# plt.savefig('p3_a.png', dpi=300)
# plt.close()

# f, ax = plt.subplots(3, 5, figsize=(10, 10))
# for i in range(5):
#     ax[0, i].imshow(X[id[i], :].reshape(28, 20), cmap='gray')
#     ax[0, i].set_title('Original')
#     ax[0, i].axis('off')
#     ax[1, i].imshow(PCA_k(X, 2)[id[i], :].reshape(28, 20), cmap='gray')
#     ax[1, i].set_title('k=2')
#     ax[2, i].imshow(PCA_k(X, 30)[id[i], :].reshape(28, 20), cmap='gray')
#     ax[2, i].set_title('k=30')

# plt.savefig('p3_b.png', dpi=300)
# plt.close()

pca = PCA()
pca.fit(X)

# Getting the cumulative variance

var_cumu = np.cumsum(pca.explained_variance_ratio_)*100

# How many PCs explain 95% of the variance?
k = np.argmax(var_cumu > 50)
print("Number of components explaining 50% variance: " + str(k))
# # print("\n")
plt.figure(figsize=[10, 5])
plt.title('Cumulative Explained Variance explained by the components')
plt.ylabel('Cumulative Explained variance')
plt.xlabel('Principal components')
plt.axvline(x=k, color="k", linestyle="--")
plt.axhline(y=50, color="r", linestyle="--")
ax = plt.plot(var_cumu)
plt.savefig('p3_c_1.png', dpi=300)


f, ax = plt.subplots(4, 5, figsize=(10, 15))
for i in range(5):
    ax[0, i].imshow(X[id[i], :].reshape(28, 20), cmap='gray')
    ax[0, i].set_title('Original')
    ax[0, i].axis('off')
    ax[1, i].imshow(PCA_k(X, 2)[id[i], :].reshape(28, 20), cmap='gray')
    ax[1, i].set_title('k=2')
    ax[2, i].imshow(PCA_k(X, 30)[id[i], :].reshape(28, 20), cmap='gray')
    ax[2, i].set_title('k=30')
    ax[3, i].imshow(PCA_k(X, 28)[id[i], :].reshape(28, 20), cmap='gray')
    ax[3, i].set_title('k=28')

plt.savefig('p3_c_2.png', dpi=300)
plt.close()
