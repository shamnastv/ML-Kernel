import numpy as np
import matplotlib.pyplot as plt


class LinearRegression():
    def __init__(self, X, y, alpha=0.03, n_iter=1500):

        self.alpha = alpha
        self.n_iter = n_iter
        self.n_samples = len(y)
        self.n_features = np.size(X, 1)
        self.X = np.hstack((np.ones(
            (self.n_samples, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))
        self.y = y[:, np.newaxis]
        self.params = np.zeros((self.n_features + 1, 1))
        self.coef_ = None
        self.intercept_ = None

    def fit(self):

        for i in range(self.n_iter):
            self.params = self.params - (self.alpha/self.n_samples) * \
            self.X.T @ (self.X @ self.params - self.y)

        self.intercept_ = self.params[0]
        self.coef_ = self.params[1:]

        return self

    def score(self, X=None, y=None):

        if X is None:
            X = self.X
        else:
            n_samples = np.size(X, 0)
            X = np.hstack((np.ones(
                (n_samples, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))

        if y is None:
            y = self.y
        else:
            y = y[:, np.newaxis]

        y_pred = X @ self.params
        score = 1 - (((y - y_pred)**2).sum() / ((y - y.mean())**2).sum())

        return score

    def predict(self, X):
        n_samples = np.size(X, 0)
        y = np.hstack((np.ones((n_samples, 1)), (X-np.mean(X, 0)) \
                            / np.std(X, 0))) @ self.params
        return y

    def get_params(self):
        return self.params


def plot_regression_line(x, x_t, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=3)

    # predicted response vector
    y_pred = b[0] + np.dot(x_t, b[1]).reshape(-1, 1)

    new_x, new_y = zip(*sorted(zip(x, y_pred)))
    # plotting the regression line
    plt.plot(new_x, new_y, color="g")
    # plt.scatter(x, y_pred, color="g", marker="o", s=3)
    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()


def transform(x, k):
    t_x = []
    for a in x:
        t_e = [0] * k
        for i in range(k):
            t_e[i] = a ** (i + 1)
        t_x.append(t_e)
    return np.array(t_x)


def main():
    # observations
    x = np.random.uniform(low=-1, high=1, size=100)
    y = np.sin(3 * x)

    x_t = transform(x, 25)

    model = LinearRegression(x_t, y, 0.05, 5000)
    model.fit()
    c = model.get_params()
    b = c[0], c[1:]
    #b = estimate_coef(x_t, y)
    print("Estimated coefficients:\nb = {}  \
    \nw = {}".format(b[0], b[1]))

    plot_regression_line(x, x_t, y, b)


if __name__ == "__main__":
    main()

