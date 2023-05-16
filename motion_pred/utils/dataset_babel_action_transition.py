import numpy as np
import os
from motion_pred.utils.dataset import Dataset
import pdb
import sys
import copy
sys.path.append("/home/llx/projects/human_pose/code/WAT/*")  
from tqdm import tqdm
import random
import torch
dtype = torch.float32
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')

class DatasetOurs(Dataset):

    def __init__(self, mode, t_his=25, t_pred=100, actions='all', use_vel=False,
                 is_6d=False, w_transi=False, **kwargs):
        self.use_vel = use_vel
        '''
        if 'acts' in kwargs.keys() and kwargs['acts'] is not None:
            self.act_name = np.array(kwargs['acts'])
        else:
            self.act_name = np.array(['stand','walk','step','stretch','sit','place something',
                                      'take_pick something up','bend','stand up','jump','throw',
                                      'kick','run','catch','wave','squat','punch','jog','kneel','hop'])
        '''
        self.act_name = np.array(['body','jump','step','turn','put_pick'])
        if 'max_len' in kwargs.keys() and kwargs['max_len'] is not None:
            self.max_len = np.array(kwargs['max_len'])
        else:
            self.max_len = 1000

        if 'min_len' in kwargs.keys() and kwargs['min_len'] is not None:
            self.min_len = np.array(kwargs['min_len'])
        else:
            self.min_len = 100
        # self.max_len = 1000
        self.mode = mode
        
        self.act_no_transi = ['sit', 'stand up']
        '''
        if 'data_file' in kwargs.keys() and kwargs['data_file'] is not None:   #1
            self.data_file = kwargs['data_file'].format(self.mode)    #1
        else:    #1
            self.data_file = os.path.join('./data', f'babel_30_300_wact_candi_{self.mode}.npz')
        '''
        self.data_file = os.path.join('../../data', f'total_data_train.npz')
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        self.traj_dim = 156
        self.normalized = False
        # iterator specific
        self.sample_ind = None
        self.w_transi = w_transi
        self.is_6d = is_6d
        if is_6d:
            self.traj_dim = self.traj_dim*2
        self.process_data()
        self.std, self.mean = None, None
        self.data_len = sum([len(seq) for seq in self.data.values()])

    def process_data(self):
        print(f'load data from {self.data_file}')
        data_o = np.load(self.data_file, allow_pickle=True)
        #pdb.set_trace()
        data_f = data_o['data'].item()
        #data_cand = data_o['data_cand'].item()

        if len(data_f.keys()) != len(self.act_name):
            # get actions of interests
            data_f_tmp = {}
            for k,v in data_f.items():
                if k not in self.act_name:
                    continue
                data_f_tmp[k] = v
            data_f = data_f_tmp

        self.data = data_f
        pass
        #self.data_cand = data_cand
        #pdb.set_trace()
        
    def sample(self, action=None, is_other_act=False, t_pre_extra=0, k=0.08, max_trans_fn=25,
               is_transi=False, n_others=1):
        if action is None:
            action = np.random.choice(self.act_name)

        max_seq_len = self.max_len.item() - self.t_his + t_pre_extra
        max_seq_len_1 = 900
        seq = self.data[action]
        #pdb.set_trace()
        # seq = dict_s[action]
        idx = np.random.randint(0, len(seq))
        # fr_end = fr_start + self.t_total
        seq = seq[idx]['poses']
        fn = [seq.shape[0]]    #900


        seq_tmp = seq#[self.t_his:]
        fn = seq_tmp.shape[0]
        print(fn)
        seq_gt = np.zeros([1, fn, seq.shape[-1]])
        #pdb.set_trace()
        seq_gt[0, :] = seq_tmp
        # seq_gt[0, fn:] = seq_tmp[-1:]
        has_nan = np.isnan(seq_gt).any()
        if has_nan:
            print('nan')
        #if np.nan in seq_gt:
            #print('nan')
        #pdb.set_trace()
        fn_gt = np.ones([1, fn])
        # fn_gt[:, fn - 1] = 1
        label_gt = np.zeros(len(self.act_name))
        # tmp = str.lower(action.split(' ')[0])
        # tmp = str.lower(action.split(' ')[0])
        label_gt[np.where(action == self.act_name)[0]] = 1
        assert np.sum(label_gt) == 1
        label_gt = label_gt[None, :]




        return  seq_gt, label_gt,fn_gt,fn

    def sample_all_act(self,action=None, is_other_act=False,t_pre_extra=0, k=0.08, max_trans_fn=25,
               is_transi=False, n_others=1):
        if action is None:
            action = np.random.choice(self.act_name)

        max_seq_len = self.max_len.item()-self.t_his + t_pre_extra
        seq = self.data[action]
        # seq = dict_s[action]
        idx = np.random.randint(0, len(seq))
        # fr_end = fr_start + self.t_total
        seq = seq[idx]['poses']
        fn = seq.shape[0]
        if fn // 10 > self.t_his:
            fr_start = np.random.randint(0, fn // 10 - self.t_his)
            seq = seq[fr_start:]
            fn = seq.shape[0]

        seq_his = seq[:self.t_his][None,:,:]
        seq_tmp = seq[self.t_his:]
        fn = seq_tmp.shape[0]
        seq_gt = np.zeros([1, max_seq_len, seq.shape[-1]])
        seq_gt[0, :fn] = seq_tmp
        seq_gt[0,fn:] = seq_tmp[-1:]
        fn_gt = np.zeros([1, max_seq_len])
        fn_gt[:, fn - 1] = 1
        fn_mask_gt = np.zeros([1, max_seq_len])
        fn_mask_gt[:, :fn+t_pre_extra] = 1
        label_gt = np.zeros(len(self.act_name))
        # tmp = str.lower(action.split(' ')[0])
        # tmp = str.lower(action.split(' ')[0])
        label_gt[np.where(action == self.act_name)[0]] = 1
        assert np.sum(label_gt) == 1
        label_gt = label_gt[None,:]

        # randomly find future sequences of other actions
        if is_other_act:
            # k = 0.08
            # max_trans_fn = 25
            seq_last = seq_his[0,-1:]
            seq_others = []
            fn_others = []
            fn_mask_others = []
            label_others = []
            #cand_seqs = self.data_cand[f'{action}_{idx}']

            # act_names = np.random.choice(self.act_name, len(self.act_name),replace=False)
            # count = 0
            for act in self.act_name:
                #cand = cand_seqs[act][:5]
                #if len(cand)<=0:
                    #continue
                for _ in range(5):
                    #cand_idx = np.random.choice(cand, 1)[0]
                    cand_idx = np.random.randint(0, 5)
                    cand_tmp = self.data[act][cand_idx]['poses']
                    cand_tmp = self.data[act][cand_idx]['poses']
                    cand_fn = cand_tmp.shape[0]
                    cand_his = cand_tmp[:max(cand_fn//10,10)]
                    dd = np.linalg.norm(cand_his-seq_last, axis=1)
                    cand_tmp = cand_tmp[np.where(dd==dd.min())[0][0]:]
                    cand_fn = cand_tmp.shape[0]
                    skip_fn = min(int(dd.min()//k + 1), max_trans_fn)
                    if cand_fn + skip_fn > max_seq_len:
                        continue
                    # cand_tmp = np.copy(cand[[-1] * (self.max_len.item()-self.t_his)])[None, :, :]
                    cand_tt = np.zeros([1, max_seq_len, seq.shape[-1]])
                    cand_tt[0, :skip_fn] = cand_tmp[:1]
                    cand_tt[0, skip_fn:cand_fn+skip_fn] = cand_tmp
                    cand_tt[0,cand_fn+skip_fn:] = cand_tmp[-1:]
                    fn_tmp = np.zeros([1, max_seq_len])
                    fn_tmp[:, cand_fn+skip_fn-1] = 1
                    fn_mask_tmp = np.zeros([1, max_seq_len])
                    fn_mask_tmp[:, skip_fn:cand_fn+skip_fn+t_pre_extra] = 1
                    cand_lab = np.zeros(len(self.act_name))
                    cand_lab[np.where(act == self.act_name)[0]] = 1
                    seq_others.append(cand_tt)
                    fn_others.append(fn_tmp)
                    fn_mask_others.append(fn_mask_tmp)
                    label_others.append(cand_lab[None,:])
                    # count += 1
                    break
                # if count == n_others:
                #     break

            if len(seq_others) > 0:
                seq_others = np.concatenate(seq_others,axis=0)
                fn_others = np.concatenate(fn_others,axis=0)
                fn_mask_others = np.concatenate(fn_mask_others,axis=0)
                label_others = np.concatenate(label_others,axis=0)

                seq_his = seq_his[[0]*(seq_others.shape[0]+1)]
                seq_gt = np.concatenate([seq_gt,seq_others], axis=0)
                fn_gt = np.concatenate([fn_gt,fn_others], axis=0)
                fn_mask_gt = np.concatenate([fn_mask_gt,fn_mask_others], axis=0)
                label_gt = np.concatenate([label_gt, label_others], axis=0)

        # randomly find sequences with transition
        if is_transi:
            if np.random.rand(1)[0] < 0.3:
                idx = np.random.randint(0,self.num_transi)
                data_transi = self.data_transi[idx]

                transi_tmp = data_transi['poses']
                sf = data_transi['frame_split'][1]-self.t_his

                seq_his_transi = transi_tmp[None,sf:sf+self.t_his]
                seq_transi_tmp = transi_tmp[None,sf+self.t_his:]
                fn_transi = seq_transi_tmp.shape[1]
                if fn_transi <= max_seq_len:
                    seq_transi_gt = np.zeros([1, max_seq_len, seq_his_transi.shape[-1]])
                    seq_transi_gt[:, :fn_transi] = seq_transi_tmp
                    seq_transi_gt[:, fn_transi:] = seq_transi_tmp[:,-1:]
                    fn_transi_gt = np.zeros([1, max_seq_len])
                    fn_transi_gt[:, fn_transi - 1] = 1
                    fn_transi_mask_gt = np.zeros([1, max_seq_len])
                    fn_transi_mask_gt[:, :fn_transi + t_pre_extra] = 1
                    label_transi_gt = np.zeros(len(self.act_name))
                    label_transi_gt[np.where(data_transi['act'][-1] == self.act_name)[0]] = 1
                    assert np.sum(label_transi_gt) == 1
                    label_transi_gt = label_transi_gt[None, :]

                    seq_his = np.concatenate([seq_his,seq_his_transi],axis=0)
                    seq_gt = np.concatenate([seq_gt, seq_transi_gt], axis=0)
                    fn_gt = np.concatenate([fn_gt, fn_transi_gt], axis=0)
                    fn_mask_gt = np.concatenate([fn_mask_gt, fn_transi_mask_gt], axis=0)
                    label_gt = np.concatenate([label_gt, label_transi_gt], axis=0)

        return seq_his,seq_gt,fn_gt,fn_mask_gt,label_gt

    def sampling_generator(self, num_samples=1000, batch_size=8,act=None,is_other_act=False,t_pre_extra=0,      #4
                           act_trans_k=0.08, max_trans_fn=25, is_transi=False,n_others=1,others_all_act=False):
        for i in range(num_samples // batch_size):
            samp_his = []
            samp_gt = []
            fn = []
            fn_mask = []
            label = []
            print(act,others_all_act)

            for i in range(batch_size):

                seq_gt, label_gt,fn_gt,fn_ = self.sample(action=act,is_other_act=is_other_act,
                                                                            t_pre_extra=t_pre_extra,
                                                                            k=act_trans_k,max_trans_fn=max_trans_fn,
                                                                            is_transi=is_transi,n_others=n_others)
                samp_gt.append(seq_gt)
                label.append(label_gt)
                fn_mask.append(fn_gt)

                # fn.append(fn_)
            samp_gt = np.concatenate(samp_gt, axis=0)
            # fn = np.concatenate(fn, axis=0)
            fn_gt = np.concatenate(fn_gt, axis=0)

            label = np.concatenate(label, axis=0)
            yield samp_gt,label, fn_gt

    def iter_generator(self, step=25):
        for data_s in self.data.values():
            for seq in data_s.values():
                seq_len = seq.shape[0]
                for i in range(0, seq_len - self.t_total, step):
                    traj = seq[None, i: i + self.t_total]
                    yield traj / 1000.



class DatasetBabel(Dataset):

    def __init__(self, mode, t_his=25, t_pred=100, actions='all', use_vel=False,
                 is_6d=False, w_transi=False, **kwargs):
        self.use_vel = use_vel
        if 'acts' in kwargs.keys() and kwargs['acts'] is not None:
            self.act_name = np.array(kwargs['acts'])
        else:
            self.act_name = np.array(['stand','walk','step','stretch','sit','place something',
                                      'take_pick something up','bend','stand up','jump','throw',
                                      'kick','run','catch','wave','squat','punch','jog','kneel','hop'])
        # if 'max_len' in kwargs.keys() and kwargs['max_len'] is not None:
        #     self.max_len = np.array(kwargs['max_len'])
        # else:
        #     self.max_len = 1000
        # self.max_len = 300
        self.max_len = int(kwargs['max_len'])

        if 'acts_class' in kwargs.keys() and kwargs['acts_class'] is not None:
            self.act_class = kwargs['acts_class']
            self.act_list_name = []
            action_class_temp = {}
            for k,v in self.act_class.items():
                print(k,v)
                self.act_list_name.append(k)
                for label_str in v:
                    frame_data = {}
                    frame_data[label_str] = k
                    action_class_temp.update(copy.deepcopy(frame_data))
            self.act_class = action_class_temp
            # print(action_class_temp)
            # exit(0)
        else:
            self.act_class = None

        if 'min_len' in kwargs.keys() and kwargs['min_len'] is not None:
            self.min_len = np.array(kwargs['min_len'])
        else:
            self.min_len = 100

        self.mode = mode
        
        self.act_no_transi = ['sit', 'stand up']

        if 'data_file' in kwargs.keys() and kwargs['data_file'] is not None:
            self.data_file = kwargs['data_file'].format(self.mode)
        else:
            self.data_file = os.path.join('./data', f'babel_30_300_wact_candi_{self.mode}.npz')

        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        self.traj_dim = 156
        self.normalized = False
        # iterator specific
        self.sample_ind = None
        self.w_transi = w_transi
        self.is_6d = is_6d
        if is_6d:
            self.traj_dim = self.traj_dim*2
        self.class_weight = {}
        self.sample_weight = []
        self.process_data()
        self.std, self.mean = None, None
        self.data_len = len(self.data)#sum([len(seq) for seq in self.data.values()])

    def process_data(self):

        print(f'load data from {self.data_file}')
        data_o = np.load(self.data_file, allow_pickle=True)
        data_f = data_o['data'].item()
        # data_cand = data_o['data_cand'].item()
        # print(self.act_name)
        # print(data_f.keys())
        # if len(data_f.keys()) != len(self.act_name):
            # get actions of interests
        data_f_tmp_ = {}
        frame_weight = {}
        for k,v in data_f.items():
            print(k,len(v))
            frame_weight["len"] = len(v)
            frame_weight["weight"] = 1/len(v)
            self.class_weight[k] = copy.deepcopy(frame_weight)
            if k not in self.act_name:
                continue
            data_f_tmp_[k] = v
        data_f = data_f_tmp_
        data_f_tmp = []
        # self.class_weight["walk"]["weight"] = self.class_weight["walk_backwards"]["weight"]/10
        # self.class_weight["kick_right"]["weight"] = self.class_weight["kick_left"]["weight"]/10

        # print(self.class_weight)
        # exit(0)

        print("Processing data...")
        for k,v in tqdm(data_f.items()):
            if k not in self.act_name:
                continue
            # print(type(v))
            # exit(0)
            frame_data = {}
            ex = 0

            for act_seq in v:
                if(len(act_seq["poses"])>self.max_len):
                    ex = ex+1
                    continue
                # print(k)
                self.sample_weight.append(copy.deepcopy(self.class_weight[k]["weight"]))
                frame_data["action_type"]=k
                frame_data["action_data"]=act_seq
                frame_data["index"] = random.randint(0,9)
                data_f_tmp.append(copy.deepcopy(frame_data))

        if self.act_class != None:
            for index in range(len(data_f_tmp)):
                action_str = data_f_tmp[index]["action_type"]
                data_f_tmp[index]["action_class"] = self.act_class[action_str]
                
        
        # print("Done!",ex)

            # data_f_tmp[k] = v

        # data_f = data_f_tmp

        self.data = data_f_tmp
        # self.data_f_tmp = data_f_tmp

    def sample(self, action=None, is_other_act=False, t_pre_extra=0, k=0.08, max_trans_fn=25,
               is_transi=False, n_others=1):
        if action is None:
            action = np.random.choice(self.act_name)

        max_seq_len = self.max_len.item() #- self.t_his + t_pre_extra
        seq = self.data[action]
        #pdb.set_trace()
        # seq = dict_s[action]
        idx = np.random.randint(0, len(seq))
        # fr_end = fr_start + self.t_total
        seq = seq[idx]['poses']
        fn = [seq.shape[0]]    #900


        seq_tmp = seq#[self.t_his:]
        fn = seq_tmp.shape[0]
        seq_gt = np.zeros([1, max_seq_len, seq.shape[-1]])
        seq_gt[0, :fn] = seq_tmp
        has_nan = np.isnan(seq_gt).any()
        if has_nan:
            print('nan')



        fn_gt = np.zeros([1, max_seq_len])
        fn_gt[:, fn - 1] = 1
        label_gt = np.zeros(len(self.act_name))
        label_gt[np.where(action == self.act_name)[0]] = 1
        assert np.sum(label_gt) == 1
        label_gt = label_gt[None, :]




        return  seq_gt, label_gt,fn_gt,fn

    def sample_single(self, index): # 参数index必写
        max_seq_len = self.max_len #- self.t_his + t_pre_extra
        seq = self.data[index]["action_data"]["poses"]

        seq_tmp = seq#[self.t_his:]
        fn = seq_tmp.shape[0]
        seq_gt = np.zeros([1, max_seq_len, seq.shape[-1]])
        seq_gt[0, :fn] = seq_tmp

        action_str = self.data[index]["action_type"]
        label_gt = np.zeros(len(self.act_name))
        label_gt[np.where(action_str == self.act_name)[0]] = 1
        fn_mask = np.zeros(max_seq_len)
        fn_mask[:fn] = 1

        fn_gt = np.zeros([1, max_seq_len])
        fn_gt[:, fn - 1] = 1
        # print(sum(fn_gt),fn)
        # print(seq_gt.shape)
        return index,action_str,label_gt ,seq_gt,fn,fn_mask,fn_gt

    def sampling_generator(self,num_samples=5000, batch_size=16):
        for i in range(num_samples // batch_size):
 
            for i in range(batch_size):
            
                 index,action_str,label_gt ,seq_gt,fn,fn_mask,fn_gt = sample_single(1)

            samp_gt = np.concatenate(samp_gt, axis=0)
            # fn = np.concatenate(fn, axis=0)
            fn_mask = np.concatenate(fn_mask, axis=0)
            label = np.concatenate(label, axis=0)

            yield samp_gt,label, fn_mask

    def iter_generator(self, step=25):
        for data_s in self.data.values():
            for seq in data_s.values():
                seq_len = seq.shape[0]
                for i in range(0, seq_len - self.t_total, step):
                    traj = seq[None, i: i + self.t_total]
                    yield traj / 1000.

    def __getitem__single_class__(self, index): # 参数index必写
        max_seq_len = self.max_len #- self.t_his + t_pre_extra
        seq = self.data[index]["action_data"]["poses"]

        seq_tmp = seq#[self.t_his:]
        fn = seq_tmp.shape[0]
        seq_gt = np.zeros([1, max_seq_len, seq.shape[-1]])
        seq_gt[0, :fn] = seq_tmp

        action_str = self.data[index]["action_type"]
        label_gt = np.zeros(len(self.act_name))
        label_gt[np.where(action_str == self.act_name)[0]] = 1
        fn_mask = np.zeros(max_seq_len)
        fn_mask[:fn] = 1

        fn_gt = np.zeros([1, max_seq_len])
        fn_gt[:, fn - 1] = 1
        # print(sum(fn_gt),fn)
        # print(seq_gt.shape)
        return index,action_str,label_gt ,seq_gt,fn,fn_mask,fn_gt
    
    def __getitem__multi_class__(self, index): # 参数index必写
        max_seq_len = self.max_len #- self.t_his + t_pre_extra
        seq = self.data[index]["action_data"]["poses"]

        seq_tmp = seq#[self.t_his:]
        fn = seq_tmp.shape[0]
        seq_gt = np.zeros([1, max_seq_len, seq.shape[-1]])
        seq_gt[0, :fn] = seq_tmp

        action_str = self.data[index]["action_type"]
        action_str_class = self.data[index]["action_class"]

        label_gt = np.zeros(len(self.act_name))
        label_gt[np.where(action_str == self.act_name)[0]] = 1

        label_gt_class = np.zeros(len(self.act_list_name))
        label_gt_class[np.where(action_str == self.act_list_name)[0]] = 1


        fn_mask = np.zeros(max_seq_len)
        fn_mask[:fn] = 1

        fn_gt = np.zeros([1, max_seq_len])
        fn_gt[:, fn - 1] = 1
        # print(sum(fn_gt),fn)
        # print(seq_gt.shape)
        return index,[action_str,action_str_class],[label_gt,label_gt_class],seq_gt,fn,fn_mask,fn_gt

    def get_render_label(self): # 参数index必写

        action_str_result = []
        action_str_class_result = []
        for action_str in self.act_name:
            action_str_class = self.act_class[action_str]

            label_gt = np.zeros(len(self.act_name))
            label_gt[np.where(action_str == self.act_name)[0]] = 1

            label_gt_class = np.zeros(len(self.act_list_name))
            label_gt_class[np.where(action_str == self.act_list_name)[0]] = 1

            action_str_result.append(copy.deepcopy(label_gt))
            action_str_class_result.append(copy.deepcopy(label_gt_class))
        action_str_result = torch.tensor(action_str_result, device=device, dtype=dtype)
        action_str_class_result = torch.tensor(action_str_class_result, device=device, dtype=dtype)

        # print(action_str_class_result.shape,action_str_result.shape)

        # print(seq_gt.shape)
        # exit(0)
        return action_str_result,action_str_class_result
    def __getitem__(self,index):
        if self.act_class==None:
            index,action_str,label_gt ,seq_gt,fn,fn_mask,fn_gt = self.__getitem__single_class__(index)
        else:
            # index,action_str,label_gt ,seq_gt,fn,fn_mask,fn_gt = self.__getitem__single_class__(index)

            index,action_str,label_gt,seq_gt,fn,fn_mask,fn_gt = self.__getitem__multi_class__(index)
        return index,action_str,label_gt ,seq_gt,fn,fn_mask,fn_gt

    def _sample_label_(self,batch_size,act=None):
        action_str_all = []
        label_gt_all = []
        if(act==None):
            sample_data = np.random.choice(self.act_name, batch_size)
        else:
            sample_data = [act]*batch_size
        for action_str in sample_data:
            # action_str = self.data[index]["action_type"]
            label_gt = np.zeros(len(self.act_name))
            label_gt[np.where(action_str == self.act_name)[0]] = 1

                # print(sum(fn_gt),fn)
                # print(fn)
            action_str_all.append(copy.deepcopy(action_str))
            label_gt_all.append(copy.deepcopy(label_gt))
        action_str_all = np.array(action_str_all)
        label_gt_all = torch.tensor(np.array(label_gt_all))

        return action_str_all,label_gt_all

    def __len__(self): 
        return self.data_len # 只需返回数据集的长度即可


if __name__ == '__main__':
    np.random.seed(0)
    # actions = {'WalkDog'}
    # generator = dataset.sampling_generator()
    # dataset.normalize_data()
    # generator = dataset.iter_generator()
    # for data, action, fn in generator:
    #     print(data.shape)
    dataset = DatasetBabel('train', actions='all', data_file='../../../data/babel_30_300_wact_candi_train.npz')
