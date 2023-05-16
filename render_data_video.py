import os
import sys
import math
import pickle
import argparse
import time
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import csv
from numba import cuda
os.environ['PYOPENGL_PLATFORM'] = 'egl' 

sys.path.append(os.getcwd())
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_ntu_act_transition import DatasetNTU
from motion_pred.utils.dataset_grab_action_transition import DatasetGrab
from motion_pred.utils.dataset_humanact12_act_transition import DatasetACT12
from motion_pred.utils.dataset_babel_action_transition import DatasetBabel
from models.motion_pred import *
from utils.fid import calculate_frechet_distance
from utils.dtw import batch_dtw_torch, batch_dtw_torch_parallel, accelerated_dtw, batch_dtw_cpu_parallel
from utils import eval_util
from utils import data_utils
from utils.vis_util import render_videos_new
from tqdm import tqdm
def get_stop_sign(Y_r,args):
    # get stop sign
    if args.stop_fn > 0:
        fn_tmp = Y_r.shape[0]
        tmp1 = np.arange(fn_tmp)[:, None]
        tmp2 = np.arange(args.stop_fn)[None, :]
        idxs = tmp1 + tmp2
        idxs[idxs > fn_tmp - 1] = fn_tmp - 1
        yr_tmp = Y_r[idxs]
        yr_mean = yr_tmp.mean(dim=1, keepdim=True)
        dr = torch.mean(torch.norm(yr_tmp - yr_mean, dim=-1), dim=1)
    else:
        dr = torch.norm(Y_r[:-1] - Y_r[1:], dim=2)
        dr = torch.cat([dr[:1, :], dr], dim=0)
    threshold = args.threshold
    tmp = dr < threshold
    idx = torch.arange(tmp.shape[0], 0, -1, device=device)[:, None]
    tmp2 = tmp * idx
    tmp2[:dataset.min_len - 1] = 0
    tmp2[-1, :] = 1
    fn = tmp2 == tmp2.max(dim=0, keepdim=True)[0]
    fn = fn.float()
    return fn

def render(epoch):

    seq_len = []
    with torch.no_grad():
        sampler = WeightedRandomSampler(torch.DoubleTensor(dataset.sample_weight), int(dataset.data_len),replacement=True)

        train_loader = DataLoader(dataset=dataset, # 要传递的数据集
                        batch_size=args.bs, #一个小批量数据的大小是多少
                        sampler=sampler,# shuffle=True, # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
                        num_workers=0) # 需要几个进程来一次性读取这个小批量数据，默认0，一般用0就够了，多了有时会出一些底层错误。
        print("Rendering dataset...")
        print(dataset.__len__())
        for index,action_str,action_label,action_seq,action_len,fn_mask,fn_gt   in tqdm(train_loader):
            # print(i,item)
            if action_seq.device!=device:
                action_seq.to(device)
            traj_tmp = action_seq.type(dtype).squeeze().contiguous()
            X = traj_tmp.cpu().numpy()
            # if cfg.dataset == 'babel':
            #     index_used = list(range(30)) + list(range(36, 66))
            #     X  = X [:, :, index_used]
            # print(X.shape)
            # exit(0)
            betas = np.zeros(10)
            for ii in range(args.bs):
                temp = X[ii]
                sequence = {'poses': X[ii][:action_len[ii].cpu().item()], 'betas': betas}
                key = f'{action_str[ii]}_{index[ii].cpu().item()}_gt'
                render_videos_new(sequence, device, cfg.data_dir, key, w_golbalrot=True, smpl_model=smpl_model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='babel_rnn')
    parser.add_argument('--cfg_classifier', default='babel_act_classifier')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=500)
    parser.add_argument('--nk', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.01)
    parser.add_argument('--stop_fn', type=int, default=5)
    parser.add_argument('--bs', type=int, default=5)
    parser.add_argument('--num_samp', type=int, default=5)
    parser.add_argument('--data_type', default='float32')
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.data_type == 'float32':
        dtype = torch.float32
    else:
        dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
        cuda.select_device(args.gpu_index)
    cfg = Config(args.cfg, test=args.test)
    cfg_classifier = Config(args.cfg_classifier, test=args.test)
    # tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
    logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

    """parameter"""
    mode = args.mode
    nz = cfg.nz
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    if 't_pre_extra' in cfg.vae_specs:
        args.t_pre_extra = cfg.vae_specs['t_pre_extra']

    """data"""

    dataset_cls = DatasetBabel
    smpl_model = 'smplh'

    # for act in cfg.vae_specs['actions']:
    dataset = dataset_cls(args.mode, t_his, t_pred, actions='all', use_vel=cfg.use_vel,
                          acts=cfg.vae_specs['actions'] if 'actions' in cfg.vae_specs else None,
                          max_len=cfg.vae_specs['max_len'] if 'max_len' in cfg.vae_specs else None,
                          min_len=cfg.vae_specs['min_len'] if 'min_len' in cfg.vae_specs else None,
                          is_6d=cfg.vae_specs['is_6d'] if 'is_6d' in cfg.vae_specs else False,
                          data_file=cfg.vae_specs['data_file'] if 'data_file' in cfg.vae_specs else None)



    render(args.iter)
