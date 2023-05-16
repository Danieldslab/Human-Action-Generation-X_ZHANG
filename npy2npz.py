import numpy as np
import os
from os.path import join as pjoin
from glob import glob
import copy


if __name__=="__main__":
    base_dir = "/home/zyf-lab/my_data/core_code/code/human_pose/code/WAT/data/HumanML3D2/joints_c"
    result_dir  = "./data/"
    save_name = pjoin(result_dir,"our_data.npz")
    action_list = ["body","jump","put_pick","step","turn"]
    npz_data = {}
    all_lable = []
    for action_str in action_list:
        action_path = pjoin(base_dir,action_str)
        files = os.listdir(action_path)
        files_dir = [f for f in files if os.path.isdir(os.path.join(action_path, f))]
        # print(files_dir)

        for dir in files_dir:
            data_dir = pjoin(action_path,dir)
            dir_action_name_list = glob(data_dir+"/*.npy")
            action_label = action_str+'_'+dir
            all_lable.append(action_label)
            npz_data.update({action_label:[]})
            for dir_action_name in dir_action_name_list:

                action_npy = np.load(dir_action_name)
                data = action_npy.copy().reshape(len(action_npy), -1, 3)
                # a = action_npy#[0]

                MINS = data.min(axis=0).min(axis=0)
                MAXS = data.max(axis=0).max(axis=0)
                colors = ['red', 'blue', 'black', 'red', 'blue',  
                        'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
                        'darkred', 'darkred','darkred','darkred','darkred']
                frame_number = data.shape[0]
                #     print(data.shape)

                height_offset = MINS[1]
                data[:, :, 1] -= height_offset
                # trajec = data[:, 0, [0, 2]]
                
                data[..., 0] -= data[:, 0:1, 0]
                data[..., 2] -= data[:, 0:1, 2]

                poses = []
                for i in range(action_npy.shape[0]):
                    pose = action_npy[i]
                    pose = pose.T
                    pose = pose.flatten()
                    poses.append(pose)
                # poses  = np.reshape(action_npy,(action_npy.shape[0],-1),order="F")
                # print(poses.shape)
                npz_data[action_label].append({"poses":np.array(poses)})
    np.savez(save_name,data=npz_data)
    print(all_lable,len(all_lable))

