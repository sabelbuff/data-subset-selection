import numpy as np
from numpy import matlib


class ADMM(object):
    def __init__(self, mu, epsilon, max_iter, reg):
        self.mu = mu
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.reg = reg

    @staticmethod
    def solveLinfshrink(y, t):
        x = np.array(y, dtype=np.float32)
        o = np.argsort(-np.absolute(y))
        z = y[o]
        cs = np.divide(np.cumsum(np.absolute(z[0:len(z) - 1])), (np.arange(1, len(z))).T) - \
             np.divide(t, np.arange(1, len(z)))
        d = np.greater(cs, np.absolute(z[1:len(z)])).astype(int)

        if np.sum(d, axis=0) == 0:
            cut_index = len(y)
        else:
            cut_index = min(np.where(d == 1)[0]) + 1

        zbar = np.mean(np.absolute(z[0:cut_index]), axis=0)

        if cut_index < len(y):
            x[o[0:cut_index]] = np.sign(z[0:cut_index]) * max(zbar - t / cut_index, np.absolute(z[cut_index]))
        else:
            x[o[0:cut_index]] = np.sign(z[0:cut_index]) * max(zbar - t / cut_index, 0)

        return x

    def optimizer1(self, C1, l, p):
        C2 = []

        if len(l) > 0:
            [D, N] = np.shape(C1)
            print(D)
            if p == 1:
                C2 = np.multiply(np.maximum(np.absolute(C1) - matlib.repmat(l, 1, N), 0), np.sign(C1))

            elif p == 2:
                # print(l.shape)
                # print(np.sqrt(np.sum(np.power(C1, 2), axis=1)).shape)
                r = np.maximum(np.sqrt(np.sum(np.power(C1, 2), axis=1, keepdims=True)) - l, 0)
                print(r.shape)
                C2 = np.multiply(matlib.repmat(np.divide(r, (r + l)), 1, N), C1)

            elif p == np.inf:
                C2 = np.zeros((D, N), dtype=np.float32)

                for i in range(D):
                    C2[i, :] = self.solveLinfshrink(C1[i, :].T, l[i]).T

        return C2

    @staticmethod
    def optimizer2(U):
        [m, N] = np.shape(U)
        V = np.flip(np.sort(U, axis=0), axis=0)
        activeSet = np.arange(0, N)
        theta = np.zeros(N)
        i = 0
        while len(activeSet) > 0 and i < m:
            j = i + 1
            temp = V[i, activeSet] - ((np.sum(V[0:j, activeSet], axis=0) - 1) / j)
            temp[temp <= 0] = 0
            temp[temp > 0] = 1
            idx = 1 - temp

            s = [k for k, t in enumerate(idx) if t > 0]

            if s:
                theta[activeSet[s]] = (np.sum(V[0:i, activeSet[s]], axis=0) - 1) / (j - 1)
            activeSet = np.delete(activeSet, s)
            i = i + 1

        if len(activeSet) > 0:

            theta[activeSet] = (np.sum(V[0:m, activeSet], axis=0) - 1) / m

        C = np.maximum((U - matlib.repmat(theta, m, 1)), 0)

        return C

    @staticmethod
    def errorCoef(Z, C):
        err = np.sum(np.sum(np.absolute(Z - C), axis=0), axis=0) / (np.size(Z , axis=0) * np.size(Z, axis=1))

        return err

    def runADMM(self, D, p):
        [Nr, Nc] = np.shape(D)
        k = 1
        idx = np.argmin(np.sum(D, axis=1))
        C1 = np.zeros((np.shape(D)))
        C1[idx, :] = 1
        Lambda = np.zeros((Nr, Nc))
        CFD = np.ones((Nr, 1))
        # p = np.inf

        while True:
            print("iteration : ", k)

            Z = self.optimizer1(np.divide(C1 - (Lambda + D), self.mu), (self.reg / self.mu) * CFD, p)
            C2 = self.optimizer2(np.divide(Z + Lambda, self.mu))
            np.set_printoptions(threshold=np.nan)

            Lambda = Lambda + np.multiply(self.mu, (Z - C2))

            err1 = self.errorCoef(Z, C2)
            err2 = self.errorCoef(C1, C2)

            if k >= self.max_iter or (err1 <= self.epsilon and err2 <= self.epsilon):
                break
            else:
                k += 1

            C1 = C2

        Z = C2

        return Z



