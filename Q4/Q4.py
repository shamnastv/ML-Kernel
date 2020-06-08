import dill
import numpy as np
import math
import matplotlib.pyplot as plt


def dist(x, y, k):
    return math.sqrt(k(x, x) + k(y, y) - 2 * k(x, y))


def dist_from_mean(x, y, k):
    m = len(y)
    temp = 0
    s_dist = k(x, x)
    for i in range(m):
        for j in range(m):
            temp += k(y[i], y[j])
    s_dist += temp / (m * m)
    temp = 0
    for i in range(m):
        temp += k(x, y[i])
    s_dist -= 2 * temp / m
    s_dist = max(0, s_dist)

    dist = math.sqrt(s_dist)
    return dist


def main():
    filename = 'kernel_4a.pkl'
    input_file = open(filename, 'rb')
    k = dill.loads(dill.load(input_file))
    d = 10
    e = np.identity(d)

    # PART A
    D = np.zeros((d, d))
    sum_D = 0
    for i in range(d):
        for j in range(d):
            D[i][j] = dist(e[i], e[j], k)
            sum_D += D[i][j]
    print('Part A. Sum = ', sum_D)

    # PART B
    D = [dist_from_mean(e[i], e, k) for i in range(d)]
    print('Part B. Sum = ', sum(D))

    # PART C
    print('Part C')
    X = np.load('data.npy')
    n = X.shape[0]

    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    c = [0] * n
    c1 = [np.random.rand(1, 2)]
    c2 = [np.random.rand(1, 2)]
    while True:

        if len(c1) == 0:
            c1 = [X[np.random.randint(0, n)]]
        if len(c2) == 0:
            c2 = [X[np.random.randint(0, n)]]

        changed = False
        for i in range(n):
            if dist_from_mean(X[i], c1, k) <= dist_from_mean(X[i], c2, k):
                if c[i] != 1:
                    changed = True
                    c[i] = 1
            else:
                if c[i] != 2:
                    changed = True
                    c[i] = 2

        if not changed:
            print('not changed')
            break
        c1 = [X[i] for i in range(n) if c[i] == 1]
        c2 = [X[i] for i in range(n) if c[i] == 2]

    c1 = np.array(c1)
    c2 = np.array(c2)
    plt.scatter(c1[:, 0], c1[:, 1], color='r')
    plt.scatter(c2[:, 0], c2[:, 1], color='g')
    plt.show()


if __name__ == '__main__':
    main()
