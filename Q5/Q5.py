import numpy as np
import matplotlib.pyplot as plt
import sys


def kernel_fn(x1, x2):
    return np.dot(x1, x2) + np.dot(x1, x2) ** 2


def get_theta(projection1, projection2):
    projection1.sort()
    projection2.sort()
    if projection1[0] > projection2[-1]:
        return (projection2[-1] + projection1[0]) / 2
    elif projection2[0] > projection1[-1]:
        return (projection1[-1] + projection2[0]) / 2
    print('Not found')
    sys.exit(0)


def main():

    # Generate Data
    n = 100
    c1 = np.random.uniform(-1, 1, n)
    c2 = np.random.uniform(-3, -1, n)
    for i in range(n):
        if c2[i] > -2:
            c2[i] = c2[i] + 4

    # Part a
    plt.scatter(c1, [0] * n, color='g')
    plt.scatter(c2, [0] * n, color='r')
    plt.show()

    # Part b
    C1 = np.zeros((n, 2))
    C2 = np.zeros((n, 2))

    C1[:, 0] = c1
    C1[:, 1] = c1 ** 2
    C2[:, 0] = c2
    C2[:, 1] = c2 ** 2

    plt.scatter(C1[:, 0], C1[:, 1], color='g')
    plt.scatter(C2[:, 0], C2[:, 1], color='r')
    plt.show()

    # Part c
    m1 = np.mean(C1, axis=0)
    m2 = np.mean(C2, axis=0)
    S1 = np.zeros((2, 2))
    S2 = np.zeros((2, 2))

    for i in range(n):
        x1 = (C1[i] - m1).reshape(-1, 1)
        x2 = (C2[i] - m2).reshape(-1, 1)
        S1 += np.dot(x1, x1.transpose())
        S2 += np.dot(x2, x2.transpose())
    SW = S1 + S2
    w = np.dot(np.linalg.inv(SW), (m2 - m1).reshape(-1, 1))

    plt.scatter(C1[:, 0], C1[:, 1], color='g')
    plt.scatter(C2[:, 0], C2[:, 1], color='r')
    plt.quiver(0, 0, w[0], w[1], color='g', scale=0.05)
    plt.show()

    # Part d
    M1 = np.zeros((2 * n, 1))
    M2 = np.zeros((2 * n, 1))
    K1 = np.zeros((2 * n, n))
    K2 = np.zeros((2 * n, n))
    I = np.eye(n)
    I1 = np.ones((n, n)) / 100
    epsilon = 0.0001
    x = []
    for i in range(n):
        x.append(c1[i])
    for i in range(n):
        x.append(c2[i])
    for i in range(2 * n):
        for j in range(n):
            M1[i] += kernel_fn(x[i], c1[j])
        M1[i] /= n
        for j in range(n):
            M2[i] += kernel_fn(x[i], c2[j])
        M2[i] /= n
    for i in range(2 * n):
        for j in range(n):
            K1[i][j] = kernel_fn(x[i], c1[j])
            K2[i][j] = kernel_fn(x[i], c2[j])

    N = np.dot(K1, np.dot((I - I1), K1.transpose())) + np.dot(K2, np.dot((I - I1), K2.transpose()))
    alpha = np.dot(np.linalg.inv(N + epsilon * np.eye((2 * n))), (M2 - M1))
    projections = []
    for i in range(2 * n):
        projection = 0
        for j in range(2 * n):
            projection += alpha[i] * kernel_fn(x[i], x[j])
        projections.append(projection[0])
    theta = get_theta(projections[:n], projections[n:])
    plt.scatter(projections[:n], [0] * n, color='g')
    plt.scatter(projections[n:], [0] * n, color='r')
    plt.scatter(theta, 0, color='b')
    plt.show()
    plt.scatter(x[:n], alpha[:n], color='g')
    plt.scatter(x[n:], alpha[n:], color='r')
    plt.show()


if __name__ == '__main__':
    main()
