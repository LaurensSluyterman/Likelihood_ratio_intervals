import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from intervals_regression_xgboost import CI_XGBx

#%% Simulating data
np.random.seed(2)
n = 40
size = (n, 1)
X = np.hstack((np.random.uniform(-1, -0.2, size=n), np.random.uniform(0.2, 1, size=n)))
Y = np.random.normal(loc=2*X**2, scale=0.1)
X = np.reshape(X, (2*n, 1))

#%% Fit and visualise a model and a perturbed model
model = xgb.XGBRegressor(objective='reg:squarederror', gamma=0.4, reg_lambda=5, base_score=np.mean(Y))
model.fit(X, Y)

# Fit a second, perturbed, model
x_star = np.array([0])
y_star = model.predict(x_star) + 1 / np.std(Y)
Y_new = np.hstack((Y, y_star))
X_new = np.hstack((X[:, 0], x_star))
X_new = np.reshape(X_new, (len(X_new), 1))
model2 = deepcopy(model)
model2.fit(X_new, Y_new)

# Visualize
x_test = np.linspace(-1.3, 1.3, 1000)
plt.figure(dpi=300)
plt.plot(X, Y, 'o')
plt.plot(x_star, y_star, 'o')
plt.plot(x_test, 2*x_test**2, linestyle='--', label=r'$f(x)$')
plt.plot(x_test, model.predict(x_test), label=r'$\hat{f}(x)$')
plt.plot(x_test, model2.predict(x_test), label=r'$\tilde{f}(x)$')
plt.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=3)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.tight_layout()
plt.show()

#%% Calculate the CIs
x_test_2 = np.linspace(-1.3, 1.3, 30)
CI = np.zeros((len(x_test_2), 2))
for i, x in enumerate(x_test_2):
    ci = CI_XGBx(model=model, x=x, X_train=X, Y_train=Y, n_steps=20, alpha=0.05, delta=1 / np.std(Y))
    CI[i, 0] = ci[0]
    CI[i, 1] = ci[1]



#%% Visualize the confidence intervals
plt.figure(dpi=200)
plt.plot(X, Y, 'o', alpha=0.2)
plt.plot(x_test, 2*x_test**2, linestyle='--', label=r'$f(x)$')
plt.plot(x_test_2, model.predict(x_test_2), label=r'$\hat{f}(x)$')
plt.fill_between(x_test_2, CI[:, 0], CI[:, 1],
                 color='blue', alpha=0.2, linewidth=0.1, label=r'CI')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.ylim((-2.6, 6.6))
plt.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=3)
plt.tight_layout()
plt.show()