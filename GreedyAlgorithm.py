import numpy as np
from random import shuffle


class Greedy(object):
    def __init__(self, dis_matrix, reg):
        self.dis_matrix = np.matrix(dis_matrix)
        self.reg = reg
        self.N = len(self.dis_matrix)

    def evaluatingFunction(self, S):
        lS = S
        ldis_matrix = self.dis_matrix
        tempdis = -ldis_matrix[:, lS]
        # print(tempdis)
        value = np.sum(np.max(tempdis, axis=1))
        value -= self.reg * len(lS)
        # print(value)
        return value

    def deterministic(self):
        X = []
        Y= np.arange(self.N)
        num = np.arange(self.N)
        # shuffle(num)
        print("Deterministic algorithm running....")
        for i in num:
            print("itreation : ", i+1)
            print(X)
            newX = X + [i]
            newY = [j for j in Y if j != i]
            if len(X) == 0:
                a = -self.evaluatingFunction(newX)
            else:
                a = self.evaluatingFunction(newX) - self.evaluatingFunction(X)
            b = self.evaluatingFunction(newY) - self.evaluatingFunction(Y)

            if a >= b:
                X = newX

            else:
                Y = newY

        function_value = self.evaluatingFunction(X)
        print(X)
        return X, function_value

    def randomized(self):
        X = []
        Y = np.arange(self.N)
        num = np.arange(self.N)
        shuffle(num)
        print("Randomized algorithm running....")
        for i in num:
            print("itreation : ", i + 1)
            newX = X + [i]
            newY = [j for j in Y if j != i]
            if len(X) == 0:
                a = -self.evaluatingFunction(newX)
            else:
                a = self.evaluatingFunction(newX) - self.evaluatingFunction(X)
            b = self.evaluatingFunction(newY) - self.evaluatingFunction(Y)

            a_dash = max(0, a)
            b_dash = max(0, b)

            values = [1, 0]

            if a_dash == 0 and b_dash == 0:
                X = newX

            else:
                a_prob = a_dash / (a_dash + b_dash)
                b_prob = 1 - a_prob
                if np.random.choice(values, p=[a_prob, b_prob]):
                    X = newX

                else:
                    Y = newY

        function_value = self.evaluatingFunction(X)

        return X, function_value
