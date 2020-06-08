import numpy as np
import dill


def get_fun(name):
    input_file = open(name, 'rb')
    return dill.loads(dill.load(input_file))


def is_valid(X, k, n):
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            k_i_j = k(X[i], X[j])
            k_j_i = k(X[j], X[i])
            if k_i_j != k_j_i:
                return False
            K[i, j] = k_i_j
            K[j, i] = k_j_i
    return np.all(np.linalg.eigvals(K) + .1 >= 0)


def main():
    n = 150
    X = []
    for i in range(n):
        x = np.random.uniform(low=-5, high=5, size=3).reshape(-1, 1)
        X.append(x)
    for i in range(4):
        filename = 'function' + str(i + 1) + '.pkl'
        func = get_fun(filename)
        print(i+1, ' : ', is_valid(X, func, n))
    sampler_file = 'k5sampler.pkl'
    filename = 'function5.pkl'
    sampler = get_fun(sampler_file)
    func = get_fun(filename)
    Y = []
    for i in range(n):
        y = sampler()
        Y.append(y)
    print(5, ' : ', is_valid(Y, func, n))


if __name__ == '__main__':
    main()
