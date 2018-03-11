import numpy as np
from random import shuffle


class Greedy(object):
    def __init__(self, dis_matrix, reg):
        self.dis_matrix = dis_matrix
        self.reg = reg
        self.N = len(self.dis_matrix)

    def evaluating_function(self, S):
        value = 0
        # print(S)
        # ar = np.zeros(self.N)
        flag = 0
        count = 0
        for i in range(self.N):
            if S[i] == 1:
                flag = 1
                count += 1
        if flag == 0:
            return 0

        for i in range(self.N):
            temp = []
            for j in range(self.N):
                if S[j] == 1:
                    # print(j)
                    temp.append(- self.dis_matrix[i, j])
            value += max(temp)
        value -= self.reg * count
        return value

    def deterministic(self):
        X = np.zeros(self.N)
        Y = np.ones(self.N)
        num = np.arange(self.N)
        shuffle(num)
        print("Deterministic algorithm running....")
        for i in range(self.N):
            print("itreation : ", i+1)
            temp_X = np.copy(X)
            temp_Y = np.copy(Y)
            X[num[i]] = 1
            Y[num[i]] = 0
            # print("for X:")
            if i == 0:
                a = -self.evaluating_function(X)
            else:
                a = self.evaluating_function(X) - self.evaluating_function(temp_X)
            # print("for Y:")
            b = self.evaluating_function(Y) - self.evaluating_function(temp_Y)
            # print(b)

            # print(a, b)

            if a > b:
                Y = temp_Y

            else:
                X = temp_X

        function_value = self.evaluating_function(X)

        return X, function_value

    def randomized(self):
        X = np.zeros(self.N)
        Y = np.ones(self.N)
        num = np.arange(self.N)
        shuffle(num)
        print("Randomized algorithm running....")
        for i in range(self.N):
            print("itreation : ", i + 1)
            temp_X = np.copy(X)
            temp_Y = np.copy(Y)
            X[num[i]] = 1
            Y[num[i]] = 0
            # print("for X:")
            if i == 0:
                a = -self.evaluating_function(X)
            else:
                a = self.evaluating_function(X) - self.evaluating_function(temp_X)
            # print("for Y:")
            b = self.evaluating_function(Y) - self.evaluating_function(temp_Y)

            # print(a, b)

            a_dash = max(0, a)
            b_dash = max(0, b)
            print(a_dash, b_dash)
            values = [1, 0]

            if a_dash == 0 and b_dash == 0:
                Y = temp_Y

            else:
                print("enter")
                a_prob = a_dash / (a_dash + b_dash)
                b_prob = 1 - a_prob
                if np.random.choice(values, p=[a_prob, b_prob]):
                    Y = temp_Y

                else:
                    X = temp_X
            print("X:", X)
            print("Y:", Y)
        function_value = self.evaluating_function(X)

        return X, function_value