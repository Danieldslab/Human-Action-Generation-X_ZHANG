import pickle
import numpy
import os

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

if __name__=="__main__":
    file_name_body = "/home/llx/projects/human_pose/code/WAT/SMPL_models/smplh/SMPLH_MALE.pkl"
    with open(file_name_body, 'rb') as smplh_file:
        smpl_body = pickle.load(smplh_file, encoding='latin1')
    file_name_hand_r = "/home/llx/projects/human_pose/code/WAT/SMPL_models/smplh/MANO_RIGHT.pkl"
    with open(file_name_hand_r, 'rb') as smplh_file:
        smpl_rhand = pickle.load(smplh_file, encoding='latin1')
    file_name_hand_l = "/home/llx/projects/human_pose/code/WAT/SMPL_models/smplh/MANO_LEFT.pkl"
    with open(file_name_hand_l, 'rb') as smplh_file:
        smpl_lhand = pickle.load(smplh_file, encoding='latin1')
    smpl_body["hands_componentsr"] = smpl_rhand["hands_components"]
    smpl_body["hands_componentsl"] = smpl_lhand["hands_components"]
    smpl_body["hands_meanr"] = smpl_rhand["hands_mean"]
    smpl_body["hands_meanl"] = smpl_lhand["hands_mean"]
    
    df2=open(file_name_body,'wb')# 注意一定要写明是wb 而不是w.
    #最关键的是这步，将内容装入打开的文件之中（内容，目标文件）
    pickle.dump(smpl_body,df2)
    df2.close()
    print("!")