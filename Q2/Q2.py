import numpy as np
import matplotlib.pyplot as plt
import cvxopt


seed = 200
tc_string = 'd'


def get_points_a():
    np.random.seed(seed)
    w_sample = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]]).reshape(2, 1)
    b_sample = np.random.normal(0, 1)
    x0_samples = np.random.uniform(-3, 3, 150).reshape(1, 150)
    x1_samples = np.random.uniform(-3, 3, 150).reshape(1, 150)
    x_samples = np.concatenate((x0_samples, x1_samples), axis=0).transpose()
    label = []
    for i in range(150):
        y = np.dot(np.transpose(w_sample), x_samples[i]) + b_sample
        if y > 0:
            label.append(1)
        else:
            label.append(-1)
    label = np.array(label)
    return x_samples[:100, :], label[:100], x_samples[100:, :], label[100:]


def get_points_c():
    np.random.seed(seed)
    x0 = np.random.uniform(-3, 3, 150).reshape(1, 150)
    x1 = np.random.uniform(-3, 3, 150).reshape(1, 150)
    x = np.concatenate((x0, x1), axis=0).transpose()
    label = []
    for i in range(150):
        y1 = x[i, 0] ** 2 + (x[i, 1] ** 2) / 2
        if y1 <= 2:
            label.append(1)
        else:
            label.append(-1)
    label = np.array(label)
    return x[:100, :], label[:100], x[100:, :], label[100:]


def draw_line(slope, intercept, label):
    axes = plt.gca()
    axes.set_xlim(axes.get_xlim())
    axes.set_ylim(axes.get_ylim())
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', label=label)


def primal_svm(C, data, label, error=1e-6, show=True):
    p1 = np.zeros((103, 103))
    p1[0, 0] = 1
    p1[1, 1] = 1
    P = cvxopt.matrix(p1, tc=tc_string)
    q1 = [0, 0, 0] + [C] * 100
    q1 = np.array(q1)
    q = cvxopt.matrix(q1, tc=tc_string)

    label = label.reshape(100, 1)
    gt = np.zeros((200, 103))
    for j in range(100):
        gt[j, 0] -= data[j, 0] * label[j]
        gt[j, 1] -= data[j, 1] * label[j]
    gt[:100, 2:3] -= label
    gt[:100, 3:103] -= np.identity(100)
    gt[100:200, 3:103] -= np.identity(100)
    G = cvxopt.matrix(gt, tc=tc_string)

    h1 = [-1] * 100 + [0] * 100
    h1 = np.array(h1)
    h = cvxopt.matrix(h1, tc=tc_string)

    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['abstol'] = error
    cvxopt.solvers.options['reltol'] = error
    cvxopt.solvers.options['feastol'] = error
    cvxopt.solvers.options['maxiters'] = 500

    sol = cvxopt.solvers.qp(P, q, G, h)
    x = sol['x']
    w = np.array([x[0], x[1]]).reshape(2, 1)
    w1, w2 = w
    b = x[2]

    predicted_label = []
    for i in range(100):
        y = np.dot(np.transpose(w), data[i]) + b
        if y > 0:
            predicted_label.append(+1)
        else:
            predicted_label.append(-1)

    count = 0
    for i in range(100):
        if label[i] == predicted_label[i]:
            count += 1
    print('Accuracy : ', count/100)
    print('C : ', C)
    print('w : ', w)
    print('b : ', b)

    if show:
        slope = -w1 / w2
        c1 = 1 - b
        c1 = c1 / w2
        plt.scatter(data[:, 0], data[:, 1], c=label)
        draw_line(slope[0], c1[0], label='wx+b=1')
        c2 = -b / w2
        draw_line(slope[0], c2[0], label='wx+b=0')
        c3 = (-1 - b) / w2
        draw_line(slope[0], c3[0], label='wx+b=-1')
        plt.legend()
        plt.show()
    return w1[0], w2[0], b


def test(test_data, test_label, w1, w2, b):
    label_pred = []
    for i in range(50):
        x1, x2 = test_data[i, :]
        y = w1 * x1 + w2 * x2 + b
        if y > 0:
            label_pred.append(+1)
        else:
            label_pred.append(-1)
    count = 0
    for i in range(50):
        if test_label[i] == label_pred[i]:
            count += 1
    print('Test Accuracy : ', count/50)


def kernel_fn(x, y, p=2):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (3e-1 ** 2)))
    # return 2 * ((1 + np.dot(x, y)) ** 2) - np.dot(x, y) ** 2 - 4 * np.dot(x, y) - 2


def kernel_svm(C, data, label):
    label = label.reshape(100, 1)
    K = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            K[i, j] = kernel_fn(data[i], data[j])

    P = cvxopt.matrix(np.outer(label, label) * K, tc=tc_string)
    q = cvxopt.matrix(np.ones(100) * -1, tc=tc_string)
    A = cvxopt.matrix(label.reshape(1, 100), tc=tc_string)
    b = cvxopt.matrix(0.0, tc=tc_string)

    tmp1 = np.diag(np.ones(100) * -1)
    tmp2 = np.identity(100)
    G = cvxopt.matrix(np.vstack((tmp1, tmp2)), tc=tc_string)

    tmp1 = np.zeros(100)
    tmp2 = np.ones(100) * C
    h = cvxopt.matrix(np.hstack((tmp1, tmp2)), tc=tc_string)

    cvxopt.solvers.options['show_progress'] = False

    sol = cvxopt.solvers.qp(P, q, G, h, A, b)

    az = np.ravel(sol['x'])

    sv1 = az > 1e-5
    ind = np.arange(len(az))[sv1]
    a = az[sv1]
    sv = data[sv1]
    sv_y = label[sv1]

    b_sum = 0.0
    for i in range(len(a)):
        b_sum += sv_y[i]
        b_sum -= np.sum(a * sv_y * K[ind[i], sv1])
    print(b_sum)
    b_sum /= len(a)

    alphas = np.array(sol['x'])
    w = np.dot((label * alphas).T, data).reshape(-1, 1)
    S = (alphas > 1e-4).flatten()
    b_sum = np.mean(label[S] - np.dot(data[S], w))

    y_predict = np.zeros(100)
    for i in range(100):
        s = 0
        for ai, sv_yi, svi in zip(a, sv_y, sv):
            s += ai * sv_yi * kernel_fn(data[i], svi)
        y_predict[i] = s
    y_predict = y_predict + b_sum
    print(b_sum)

    count = 0
    label_pred = []
    for i in range(100):
        if y_predict[i] > 0:
            label_pred.append(1)
        else:
            label_pred.append(-1)

    for i in range(100):
        if label_pred[i] == label[i]:
            count += 1
    print('Accuracy', count/100)
    return b_sum, a, sv_y, sv


def main():
    print('Part A')
    train_data, train_label, test_data, test_label = get_points_a()
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_label)
    plt.plot()
    plt.show()

    # Part B
    print('Part B')
    C = 2
    w1, w2, b = primal_svm(C, train_data, train_label,  show=True)
    test(test_data, test_label, w1, w2, b)

    # Part C
    print('Part C')
    train_data, train_label, test_data, test_label = get_points_c()
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_label)
    plt.show()

    C = 2
    w1, w2, b = primal_svm(C, train_data, train_label, show=False)
    test(test_data, test_label, w1, w2, b)

    # Part D
    print('Part D')
    train_data_d = np.square(train_data)
    test_data_d = np.square(test_data)
    plt.scatter(train_data_d[:, 0], train_data_d[:, 1], c=train_label)
    plt.show()

    C = 1
    w1, w2, b = primal_svm(C, train_data_d, train_label, show=True)
    test(test_data_d, test_label, w1, w2, b)

    # Part E
    print('Part E')
    C = 1
    y_predict = np.zeros(100)
    b, a, sv_y, sv = kernel_svm(C, train_data, train_label)
    for i in range(50):
        s = 0
        for al, sv_yl, svl in zip(a, sv_y, sv):
            s += al * sv_yl * kernel_fn(test_data[i], svl)
        y_predict[i] = s
    y_predict = y_predict + b
    correct_label = 0
    label_pred = []
    for i in range(50):
        if y_predict[i] > 0:
            label_pred.append(1)
        else:
            label_pred.append(-1)

    for i in range(50):
        if label_pred[i] == test_label[i]:
            correct_label += 1
    print('Test Accuracy : ', correct_label/50)


if __name__ == '__main__':
    main()
