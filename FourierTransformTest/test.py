import numpy as np
import matplotlib.pyplot as plt

X = np.arange(100).reshape(-1, 1)
y = 50*np.sin(np.arange(100)) + np.arange(100) * 3

coef = (np.linalg.inv(X.T @ X) @ X.T @ y)[0]

plt.plot(X, y, label="y")
plt.plot(np.arange(100), y - coef * np.arange(100), label=f"y + {coef:.2f}*x")
plt.legend()
plt.show()



