import numpy as np
import os


if __name__=="__main__":
    data_o = np.load('../../data/total_data_train.npz', allow_pickle=True)
    data_f = data_o['data'].item()
    print("Done")