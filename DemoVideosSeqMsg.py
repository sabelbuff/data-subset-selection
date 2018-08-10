"""
This file runs the demo for "subset selection in sequential data using message passing"

To know more about this, please read :
Subset Selection and Summarization in Sequential Data
by Ehsan Elhamifar, M. Clara De Paolis Kaluza
http://www.ccs.neu.edu/home/eelhami/publications/SeqSS-NIPS17-Ehsan.pdf

Install 'hmmlearn' python library to run this demo using script:

            pip install hmmlearn

"""

import numpy as np
import os
import scipy.io as io
from hmmlearn import hmm
from DS3 import DS3
from EvaluateSummary import evaluateSummarySuperframe
import sys
import time

np.set_printoptions(threshold=np.nan)
np.random.seed(42)

# get the current path.
cwd = os.getcwd()

# folders containing video data(.mat) file(superframes values, user_scores, ....).
sum_me_dir = cwd + "/SumMeFinal/"
tv_sum_gt = cwd + "/SumMe_GT/"

# hmm model arguement.
lengths = []
i = 0

for root, dirs, files in os.walk(sum_me_dir):
    for file in files:
        # get the first video data file.
        if i == 0:
            filename = os.path.join(root, file)
            filename1 = os.path.join(tv_sum_gt, file)
            vid_str = io.loadmat(filename)

            # superframe features.
            X = np.array(vid_str['vid_str']['c3d_fc6'][0][0])
            Y = X

            # superframe indices.
            superframe = np.array(vid_str['vid_str']['superframe'][0][0], dtype=np.double)
            ground_truth = io.loadmat(filename1)

            # user_score for the video superframes.
            user_score = np.array(ground_truth['user_score'], dtype=np.double)
            lengths.append(len(X))
            i = i+1

        # rest of the videos.
        else:
            filename = os.path.join(root, file)
            vid_str = io.loadmat(filename)
            X1 = np.array(vid_str['vid_str']['c3d_fc6'][0][0])
            lengths.append(len(X1))
            X = np.concatenate((X1, X), axis=0)

# number of HMM states or the states present in the video.
num_of_states = 30
model = hmm.GaussianHMM(n_components=num_of_states)
model.fit(X, lengths)

# value of the states after training.
states = model.means_

# transition probability of the states.
state_trans_prob = model.transmat_

# initial probability of the states.
state_init_prob = model.startprob_

# video data for which subset is to be found.
target_video = Y

M = len(states)
N = len(target_video)

dis_matrix = np.zeros((M, N))

# calculate dis-similarity matrix between source(states) and target(video superframes)
for i in range(M):
    for j in range(N):
        dis_matrix[i, j] = np.linalg.norm((states[i] - target_video[j]), 2)

# initialize DS3 class with dis-similarity matrix and the regularization parameter.
reg = 5
DS = DS3(dis_matrix, reg)

# run the message passing algorithm.
start = time.time()
data_msg_seq, num_rep_msg_seq, obj_func_value_msg_seq = \
    DS.messagePassingSeq(damp=0.6, max_iter=2000, trans_matrix=state_trans_prob, init_prob_matrix=state_init_prob)
end = time.time()

# representative states found by running the above algorithm.
rep_state = data_msg_seq

rep_super_frames = []

# find the superframes which are representatives by taking the superframes which are
# closest(euclidean distance) to the representative states.
for i in rep_state:
    min = sys.maxsize
    min_index = 0
    for j in range(len(target_video)):
        dist = np.linalg.norm((states[i] - target_video[j]), 2)
        if min > dist:
            min = dist
            min_index = j

    rep_super_frames.append(min_index)

# change the above indices into 0s and 1s for all indices.
summary = np.zeros(len(target_video))
for i in range(len(rep_super_frames)):
    summary[rep_super_frames[i]] = 1

run_time = end - start

obj_func_value = obj_func_value_msg_seq

# calculate precision, recall, and f-score.
recall, precision, f_score = evaluateSummarySuperframe(summary, user_score, superframe)

print("Run Time :", run_time)
print("Objective Function Value  :", obj_func_value)
print("Recall :", recall)
print("Precision :", precision)
print("F-Score :", f_score)