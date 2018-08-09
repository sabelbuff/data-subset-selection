import numpy as np


"""
    This function evaluates a summary(f-score, precision, recall) for a given subset selection of a video
    
    f_measure is the mean pairwise f-measure used in Gygli et al. ECCV 2014
    
    :param summary_selection:       selected representatives(superframes) of the video.
    :param user_score:              score given by users for representative selection.
    :param super_frame_index:       indexing of the video by superframes.            

    :returns:                       precision, recall, f-score. 
    """


def evaluateSummarySuperframe(summary_selection, user_score, super_frame_index):
    f_score_beta = 1
    num_frames = super_frame_index[-1]
    num_users = len(user_score)

    # Convert user summaries to superframe representation.
    # user_score has human-generated summaries frame-by-frame.
    # convert this by binning the frames into their corersponding superframe.
    # user_superframe will be a nSuperFrames x nbOfUsers array where element (i,j) will be the number
    # of frames within superframe i that user j included in their summary.
    num_super_frames = len(super_frame_index)
    user_score_sf = np.zeros(num_super_frames, num_users)

    for sf in range(num_super_frames):
        user_score_sf[sf,:] = np.sum(np.where
                                     (user_score[super_frame_index[sf, 0]: super_frame_index[sf, 1],:] == 0, 0, 1))


    # Convert auto summary from superframes to frames.
    # We say that if superframe i was chosen in the summary, then all frames
    # that are within that superframe are selected.
    auto_summary_by_frame = np.zeros(num_frames, 1)
    sf_in_summary = np.where(summary_selection == 0, 0, 1)

    for sf_indx in range(len(sf_in_summary)):
        sf = sf_in_summary[sf_indx]
        auto_summary_by_frame[super_frame_index[sf, 0]: super_frame_index[sf, 1]] = 1

    gt_summaries = np.where(user_score == 0, 0, 1)

    user_intersection = np.zeros(num_users)

    # Compute (pairwise) f-measure, precision and recall.
    sensitivity = np.zeros(num_users)
    specificity = np.zeros(num_users)
    precision = np.zeros(num_users)
    recall = np.zeros(num_users)
    auto_summary = auto_summary_by_frame
    N = num_frames

    for userIdx in range(num_users):
        overlap = np.multiply(gt_summaries[:, userIdx], auto_summary)
        TP = np.sum(np.where(overlap == 0, 0, 1))
        FP = np.sum(auto_summary - np.where(overlap == 0, 0, 1))
        TN = np.sum(np.multiply(np.where(gt_summaries[:, userIdx] == 0, 1, 0), np.where(auto_summary == 0, 1, 0)))
        FN = (N - np.sum(auto_summary)) - TN
        sensitivity[userIdx] = TP / (TP + FN)
        specificity[userIdx] = TN / (FP + TN)
        precision[userIdx] = TP / (TP + FP)         # precision
        recall[userIdx] = TP / (TP + FN)
        user_intersection[userIdx] = np.sum(overlap)

    recall = np.divide(user_intersection[0:num_users], np.sum(np.where(gt_summaries[:, 0:num_users] > 0, 1, 0)))
    f_measure = np.divide(((1 + f_score_beta ^ 2) * (np.multiply(recall, precision))),
                          ((f_score_beta ^ 2) * (precision + recall)))
    f_measure[np.isnan(f_measure)] = 0

    return recall, precision, f_measure