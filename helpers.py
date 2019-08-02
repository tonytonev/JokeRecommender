import time
import numpy as np

# Convert data to make Y a (num_jokes x num_users) matrix with their rating between
# 0 and 1. R is a binary-valued indicator matrix. 1 if rated, 0 if not rated.
# ID's become zero-indexed so we don't waste space, and id's in the data set start from 1
def reshape_data(ratings, print_prog=True):
    n_users = len(ratings.user_id.unique())
    n_jokes = len(ratings.joke_id.unique())

    Y = np.zeros((n_jokes, n_users))
    R = np.zeros((n_jokes, n_users))

    start_time = time.time()
    last_print = start_time

    for i, (index, row) in enumerate(ratings.iterrows()):
        Y[row.joke_id - 1][row.user_id - 1] = (row.Rating + 10.) / 20.
        R[row.joke_id - 1][row.user_id - 1] = 1

        if print_prog:
            last_print = print_progress(i, len(ratings), start_time, last_print, 'Reshaping data')

    if print_prog:
        print("100%% finished reshape in %.1f seconds" % (time.time() - start_time))

    return R, Y

# Just like reshape_data(), but only for R
def reshape_data_R(df, print_prog=True):
    n_users = len(df.user_id.unique())
    n_jokes = len(df.joke_id.unique())

    R = np.zeros((n_jokes, n_users))

    start_time = time.time()
    last_print = start_time

    for i, (index, row) in enumerate(df.iterrows()):
        R[row.joke_id - 1][row.user_id - 1] = 1

        if print_prog:
            last_print = print_progress(i, len(df), start_time, last_print, 'Reshaping data')

    if print_prog:
        print("100%% finished reshape in %.1f seconds" % (time.time() - start_time))

    return R

# Writes the predictions in the expected csv format for the competition
def write_predictions(prediction_matrix, test_R, file_name):
    start_time = time.time()
    last_print = start_time

    f = open(file_name, 'w')

    f.write('id,Rating\n')

    it = np.nditer(test_R, flags=['multi_index'])
    i = 0
    while not it.finished:
        if it[0] == 1:
            joke_id, user_id = it.multi_index
            f.write('%s_%s,%f\n' % (user_id + 1, joke_id + 1, prediction_matrix[joke_id, user_id]))

            last_print = print_progress(i, 537880, start_time, last_print, 'Writing predictions')

            i += 1

        it.iternext()
    print("100%% finished writing predictions in %.1f seconds" % (time.time() - start_time))
    f.close()

def print_progress(curr, total, start_time, last_print, str=''):
    # Every second, print update
    if time.time() - last_print > 1:
        time_spent = time.time() - start_time
        time_remaining = time_spent / (curr / total) - time_spent
        percent_done = curr / total * 100

        if (str):
            print("%s -- " % str, end='')
        if (time_remaining > 60):
            print("%.1f%% %.0f minutes remaining" % (percent_done, time_remaining/60))
        else:
            print("%.1f%% %.0f seconds remaining" % (percent_done, time_remaining))
        last_print = time.time()

    return last_print