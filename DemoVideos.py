import numpy as np
import os
import scipy.io as io
from hmmlearn import hmm
from DS3 import DS3
from EvaluateSummary import evaluateSummarySuperframe

np.set_printoptions(threshold=np.nan)
np.random.seed(42)

cwd = os.getcwd()
sum_me_dir = cwd + "/SumMeFinal/"

lengths = []
i = 0
for root, dirs, files in os.walk(sum_me_dir):
    for file in files:
        if i == 0:
            filename = os.path.join(root, file)
            vid_str = io.loadmat(filename)
            X = np.array(vid_str['vid_str']['c3d_fc6'][0][0])
            Y = X
            lengths.append(len(X))
            i = i+1
        else:
            filename = os.path.join(root, file)
            vid_str = io.loadmat(filename)
            X1 = np.array(vid_str['vid_str']['c3d_fc6'][0][0])
            lengths.append(len(X1))
            X = np.concatenate((X1, X), axis=0)

num_of_states = 30
model = hmm.GaussianHMM(n_components=num_of_states)
model.fit(X, lengths)

states = model.means_
state_trans_prob = model.transmat_
state_init_prob = model.startprob_

target_video = Y

M = len(states)
N = len(target_video)

n_samples = 50
X = np.random.uniform(0, 1, size=(n_samples, 2))
M = n_samples
N = n_samples
dis_matrix = np.zeros((M, N))

for i in range(M):
    for j in range(N):
        # dis_matrix[i, j] = np.linalg.norm((states[i] - target_video[j]), 2)
        dis_matrix[i, j] = np.linalg.norm((X[i] - X[j]), 2)

print(dis_matrix)

reg = 10
DS = DS3(dis_matrix, reg)

# data_msg_seq, num_rep_msg_seq, obj_func_value_msg_seq = \
#     DS.messagePassingSeq(damp=0.6, max_iter=1000, trans_matrix=state_trans_prob, init_prob_matrix=state_init_prob)
#
# print(data_msg_seq)

data_msg, num_rep_msg, obj_func_value_msg = DS.messagePassing(damp=0.6, max_iter=1000)

print(data_msg)

data_admm, num_of_rep_admm, obj_func_value_admm, obj_func_value_post_proc_admm = \
    DS.ADMM(mu=10 ** -1, epsilon=10 ** -7, max_iter=200000, p=np.inf)

print(data_admm)
#
# data_greedy, num_of_rep_greedy, obj_func_value_greedy = DS.greedyDeterministic()
#
# print(data_greedy)



# summary = np.zeros(len(states))
#     for i in range(len(D8)):
#         summary[D8[i]] = 1