import numpy as np
import matplotlib.pyplot as plt


def get_coeff(x, y):
    n = x.shape[0]

    ones = np.ones(shape=n).reshape(-1, 1)
    x = np.concatenate((ones, x), 1)
    c = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)

    b = c[0]
    w = c[1:]

    return b, w


def prediction(x, x_t, y, b):
    plt.scatter(x, y, color='m', marker='o', s=3)

    y_pred = b[0] + np.dot(x_t, b[1]).reshape(-1, 1)

    mse = ((y.reshape(-1) - y_pred.reshape(-1)) ** 2).mean()
    print('mse train ', mse)

    new_x, new_y = zip(*sorted(zip(x, y_pred)))
    plt.plot(new_x, new_y, color='g')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def kernel_fn(x_i, x_j):
    t = np.dot(x_i, x_j)
    return (1 + t) ** 10
    # return np.tanh(t + .8)


def pred_kernel(x, y, x_test, y_test):
    a = 1
    n = x.shape[0]
    m = x_test.shape[0]

    K = np.zeros(shape=(n, n))
    K_test = np.zeros(shape=(n, m))
    for i in range(n):
        for j in range(i, n):
            k = kernel_fn(x[i], x[j])
            K[i, j] = k
            K[j, i] = k
        for j in range(m):
            K_test[i, j] = kernel_fn(x[i], x_test[j])

    y_pred_train = np.dot(K.transpose(), (np.dot(np.linalg.inv(K + a * np.identity(n)), y)))
    y_pred_test = np.dot(K_test.transpose(), (np.dot(np.linalg.inv(K + a * np.identity(n)), y)))

    mse_train = ((y.reshape(-1) - y_pred_train.reshape(-1)) ** 2).mean()
    mse_test = ((y_test.reshape(-1) - y_pred_test.reshape(-1)) ** 2).mean()
    print('mse train ', mse_train)
    print('mse test ', mse_test)

    # Plotting Train
    plt.scatter(x, y, color='m', marker='o', s=3)
    plt.scatter(x, y_pred_train, color='g', marker='o', s=3)

    #Plotting Function
    new_x, new_y = zip(*sorted(zip(x, y_pred_train)))
    plt.plot(new_x, new_y, color='g')

    plt.xlabel('x')
    plt.ylabel('y')

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
    # Data Generation
    x = np.random.uniform(low=-1, high=1, size=100)
    y = np.sin(3 * x)

    # Part a
    print('\n============Part A===========')
    plt.scatter(x, y, color='m', marker='o', s=3)
    plt.show()

    # Part b
    print('\n============Part B===========')
    x_t = transform(x, 1)
    b, w = get_coeff(x_t, y)
    prediction(x, x_t, y, (b, w))

    # Part c
    print('\n============Part C===========')
    for k in range(10):
        x_t = transform(x, k+1)

        b, w = get_coeff(x_t, y)
        print('k = ', k + 1,)
        prediction(x, x_t, y, (b, w))

    # Part d
    print('\n============Part D===========')
    x_test = np.random.uniform(low=-1, high=1, size=100)
    y_test = np.sin(3 * x_test)

    pred_kernel(x, y, x_test, y_test)


if __name__ == '__main__':
    main()
