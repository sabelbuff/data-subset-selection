"""
This file runs the demo for "subset selection of data using greedy algorithm"

To know more about this, please read :
A TIGHT LINEAR TIME (1/2)-APPROXIMATION FOR UNCONSTRAINED SUBMODULAR MAXIMIZATION
by NIV BUCHBINDER, MORAN FELDMAN, JOSEPH (SEFFI) NAOR, AND ROY SCHWARTZ
https://www.openu.ac.il/personal_sites/moran-feldman/publications/SICOMP2015.pdf

"""

import numpy as np
import os
import scipy.io as io
from DS3 import DS3
from EvaluateSummary import evaluateSummarySuperframe
import time

np.set_printoptions(threshold=np.nan)
np.random.seed(42)

# get the current path.
cwd = os.getcwd()

# folders containing video data(.mat) file(superframes values, user_scores, ....).
sum_me_dir = cwd + "/SumMeFinal/"
tv_sum_gt = cwd + "/SumMe_GT/"

for root, dirs, files in os.walk(sum_me_dir):
    for file in files:
        # get the first video data file.
        filename = os.path.join(root, file)
        filename1 = os.path.join(tv_sum_gt, file)
        vid_str = io.loadmat(filename)

        # superframe features.
        X = np.array(vid_str['vid_str']['c3d_fc6'][0][0])

        # superframe indices.
        superframe = np.array(vid_str['vid_str']['superframe'][0][0], dtype=np.double)
        ground_truth = io.loadmat(filename1)

        # user_score for the video superframes.
        user_score = np.array(ground_truth['user_score'], dtype=np.double)
        break


N = len(X)

# calculate dis-similarity matrix between source(states) and target(video superframes).
dis_matrix = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        dis_matrix[i, j] = np.linalg.norm((X[i] - X[j]), 2)

print(dis_matrix)

# initialize DS3 class with dis-similarity matrix and the regularization parameter.
reg = 5
DS = DS3(dis_matrix, reg)

# run the greedy algorithm.
start = time.time()
data_greedy, num_of_rep_greedy, obj_func_value_greedy = DS.greedyDeterministic()
end = time.time()

rep_super_frames = data_greedy

# change the above indices into 0s and 1s for all indices.
summary = np.zeros(N)
for i in range(len(rep_super_frames)):
    summary[rep_super_frames[i]] = 1

run_time = end - start

obj_func_value = obj_func_value_greedy

# calculate precision, recall, and f-score.
recall, precision, f_score = evaluateSummarySuperframe(summary, user_score, superframe)

print("Run Time :", run_time)
print("Objective Function Value  :", obj_func_value)
print("Recall :", recall)
print("Precision :", precision)
print("F-Score :", f_score)
