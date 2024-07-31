import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

np.random.seed(6)


# We want to approximate this function
def generate_linear_function(n_features):
    coefficient = (2 * np.random.rand(n_features) - 0.5)
    bias = np.random.rand()

    class Foo:
        def __init__(self):
            self.coefficients = coefficient
            self.bias = bias

        def foo(self, x):
            n_features = len(x[0])
            return np.sum(self.coefficients[:n_features] * x, axis=1) + self.bias

        def __call__(self, x):
            return self.foo(x)

    return Foo()


def generate_noise_fun(mean=0, std=5):
    def foo(N=1):
        if N == 1:
            return np.random.normal(mean, std)
        else:
            return np.random.normal(mean, std, N)

    return foo


def generate_data(linear_fun, noise_fun, N, n_features):
    X = np.random.rand(N, n_features) * 100
    y = X[:, -1] + noise_fun(N) > linear_fun(X[:, :-1])
    return X, y


def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    """Gaussian Kernel"""
    dist = cdist(X1, X2, metric="sqeuclidean")
    return sigma_f ** 2 * np.exp(-0.5 / length_scale ** 2 * dist)


class GaussianProcess:
    def __init__(self, kernel=rbf_kernel, noise=1e-10):
        self.kernel = kernel
        self.noise = noise

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.K = self.kernel(X_train, X_train) + self.noise * np.eye(len(X_train))

    def predict(self, X_s):
        K_s = self.kernel(self.X_train, X_s)
        K_ss = self.kernel(X_s, X_s) + self.noise * np.eye(len(X_s))
        K_inv = np.linalg.inv(self.K)

        mu_s = K_s.T @ K_inv @ self.Y_train
        cov_s = K_ss - K_s.T @ K_s

        return mu_s, cov_s

    def sample(self, X_s, n_samples=1):
        mu_s, cov_s = self.predict(X_s)
        return np.random.multivariate_normal(mu_s.ravel(), cov_s, n_samples)


def expected_improvement(X_sample, Y_sample, gpr, xi=0.01):
    mu, sigma = gpr.predict(X_sample)
    mu_sample_opt = np.max(Y_sample)

    sigma = sigma.reshape(-1, 1)

    with np.errstate(divide="warn"):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None

    def min_obj(X):
        return -acquisition(X.reshape(-1, dim), Y_sample, gpr)

    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method="L-BFGS-B")
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x
    return min_x.reshape(-1, 1)


linear_fun = generate_linear_function(2)
noise_fun = generate_noise_fun(mean=0, std=30)
X, y = generate_data(linear_fun, noise_fun, 2000, 2)
gpc = GaussianProcess()


def objective(C, max_iter):
    model = LogisticRegression()
    model.set_params(C=C, max_iter=int(max_iter))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return -np.mean(cross_val_score(model, X, y, cv=5, n_jobs=1, scoring="accuracy"))


# Initial gaussian process value
GX = np.array([[1e-3, 50], [1e-1, 10], [1e1, 100], [1.0, 100]])
GY = np.array([objective(C, max_iter) for C, max_iter in GX])

# Initialize Gaussian process
gpc.fit(GX, GY)
for i in range(30):
    X_next = propose_location(expected_improvement, GX, GY, gpc, np.array([[1e-6, 1], [10, 100]]))
    Y_next = objective(X_next[0][0], int(X_next[1][0]))

    GX = np.vstack((GX, X_next.T))
    GY = np.append(GY, Y_next)

    gpc.fit(GX, GY)

best_C, best_max_it = GX[GY.argmax()][0], int(GX[GY.argmax()][1])
model = LogisticRegression()
model.set_params(C=best_C, max_iter=int(best_max_it))
model.fit(X, y)
print("Best model: ", model)

# Create a mesh grid
x_min, x_max = 0, 100 + 1
y_min, y_max = 0, 100 + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                     np.arange(y_min, y_max, 1))
# Predict probabilities for each point in the mesh grid
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 10))
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.contourf(xx, yy, Z, alpha=0.8, levels=np.linspace(0, 1, 100))
plt.scatter(X[y == 0, 0], X[y == 0, 1], label="class 1", color="orange")
plt.scatter(X[y == 1, 0], X[y == 1, 1], label="class 2", color="blue")
x_min, x_max = X[:, 0].min(), X[:, 0].max()
plt.plot(np.linspace(0, 100), np.linspace(0, 100) * linear_fun.coefficients[0] + linear_fun.bias, label="True function",
         linestyle="dashed", color="white")
plt.legend()
plt.savefig("plot.png", dpi=300)
