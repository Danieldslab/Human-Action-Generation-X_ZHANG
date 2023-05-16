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
from torch.utils.data import DataLoader
import copy
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

def val(epoch):

    seq_len = []
    with torch.no_grad():

        st = time.time()

        train_loader = DataLoader(dataset=dataset, # 要传递的数据集
                                batch_size=cfg.batch_size, #一个小批量数据的大小是多少
                                shuffle=True, # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
                                num_workers=0) # 需要几个进程来一次性读取这个小批量数据，默认0，一般用0就够了，多了有时会出一些底层错误。
        
        for index,action_str,action_label,action_seq,action_len,fn_mask,fn_gt  in train_loader:
            # traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
            # traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
            if action_seq.device!=device:
                action_seq = action_seq.to(device)

            traj_tmp = action_seq.type(dtype).squeeze().contiguous() 
            X = traj_tmp.clone()
            if(isinstance (action_label,list)):
                action_calss_label= action_label[1].clone()
                action_label = action_label[0].clone()
                if action_label.device!=device:
                    action_label = action_label.to(device)       
                action_label = action_label.type(dtype)
                if action_calss_label.device!=device:
                    action_calss_label = action_calss_label.to(device)       
                action_calss_label = action_calss_label.type(dtype)
            # if action_label.device!=device:
            #     action_label = action_label.to(device)       
            # action_label = action_label.type(dtype)
            # print(action_label.shape,action_str.shape)
            # exit(0)
            # fn = tensor(fn, device=device, dtype=dtype)
            if fn_mask.device!=device:
                fn_mask = fn_mask.to(device)
            # print(fn_mask.shape,traj_tmp.shape)
            fn_mask = fn_mask.type(dtype).squeeze()   

            if fn_gt.device!=device:
                fn_gt = fn_gt.to(device)
            # print(fn_mask.shape,traj_tmp.shape)
            fn_gt = fn_gt.type(dtype).squeeze()  
            Y = traj_tmp.permute(1, 0, 2)#[t_his:]
            
            # print(fn_mask.shape,Y.shape)
            if cfg.dataset == 'babel':
                index_used = list(range(30)) + list(range(36, 66))
                Y = Y[:, :, index_used]

            # Y_r, mu, logvar, pmu, plogvar = model( Y, action_label, fn_gt)
            # sample_action_str,sample_action_label = dataset._sample_label_(cfg.batch_size)
            # if sample_action_label.device!=device:
            #     sample_action_label = sample_action_label.to(device)       
            # sample_action_label = sample_action_label.type(dtype)
            # label = torch.eye(cfg.vae_specs['n_action'], device=device, dtype=dtype)
            label,label_class = dataset.get_render_label()
            sample_action_str = cfg.vae_specs['actions']*(cfg.batch_size//cfg.vae_specs['n_action'])
            sample_action_label = label.repeat([cfg.batch_size//cfg.vae_specs['n_action'],1])
            sample_action_label_class = label_class.repeat([cfg.batch_size//cfg.vae_specs['n_action'],1])

            # sample_action_str = sample_action_str*(cfg.batch_size//cfg.vae_specs['n_action'])
            # print(sample_action_str)
            if(sample_action_label.shape[0]<cfg.batch_size):
                sample_index = int(cfg.batch_size-sample_action_label.shape[0])
                sample_action_label = torch.cat((sample_action_label,sample_action_label[:sample_index,:]))
                sample_action_label_class = torch.cat((sample_action_label_class,sample_action_label_class[:sample_index,:]))
                sample_action_str = sample_action_str+sample_action_str[:sample_index]
            # print(sample_action_label.shape,len(sample_action_str))
            # exit(0)
            our_method = True
            if(our_method):
                sample_action_label = [sample_action_label,sample_action_label_class]
            Y_r_sample = model.sample_prior(sample_action_label)
            # Y_r = Y_r.permute(1, 0, 2)
            Y_r = Y_r_sample.permute(1, 0, 2)
            # print(X.shape,Y_r.shape)
                # fn = get_stop_sign(Y_r,args)
                # seq_l = torch.where(fn[cfg.t_his:].transpose(0, 1) == 1)[1].cpu().data.numpy()+1
                # seq_len.append(seq_l)
                # seq_l = seq_l.reshape([-1, args.nk])
                # seq_l = torch.where(fn.transpose(0, 1) == 1)[1].cpu().data.numpy() + 1
                # seq_l = seq_l.reshape([bs, cfg.vae_specs['n_action'], args.nk])

            # print(Y_r.shape,x.shape)
            # Y_r = x
            if cfg.dataset == 'babel':
                traj_tmp = torch.clone(X)
                index_used = list(range(30)) + list(range(36, 66))
                traj_tmp[:, :, index_used] = Y_r
                Y_r = traj_tmp.clone()

                traj_tmp = torch.clone(X)
                index_used = list(range(30)) + list(range(36, 66))
                traj_tmp[:, :, index_used] = Y_r_sample.permute(1, 0, 2)
                Y_r_sample = traj_tmp.clone()
            # print(Y_r.shape,x.shape)
            # y = Y_r.reshape([-1,bs, cfg.vae_specs['n_action'], args.nk,Y_r.shape[-1]]).cpu().data.numpy()
            betas = np.zeros(10)
            X = X.cpu().numpy()
            Y_r = Y_r.cpu().numpy()
            Y_r_sample = Y_r_sample.cpu().numpy()

            for ii in range(cfg.batch_size):
                # sequence = {'poses': X[ii][:action_len[ii].cpu().item()], 'betas': betas}
                # key = f'{action_str[ii]}_{index[ii].cpu().item()}_gt'
                # render_videos_new(sequence, device, cfg.result_dir + f'/{args.mode}', key, w_golbalrot=True, smpl_model=smpl_model)
                # sequence = {'poses': Y_r[ii], 'betas': betas}
                # key = f'{action_str[ii]}_{index[ii].cpu().item()}_test'
                # render_videos_new(sequence, device, cfg.result_dir + f'/{args.mode}', key, w_golbalrot=True, smpl_model=smpl_model)

                sequence = {'poses': Y_r_sample[ii], 'betas': betas}
                key = f'sample_{sample_action_str[ii]}_{index[ii].cpu().item()}'
                render_videos_new(sequence, device, cfg.result_dir + f'/{str(args.iter)}', key, w_golbalrot=True, smpl_model=smpl_model)
        # print(f">>>> action {act} time used {time.time()-st:.3f}")

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
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
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
    if cfg.dataset == 'grab':
        dataset_cls = DatasetGrab
        smpl_model = 'smplx'
    elif cfg.dataset == 'ntu':
        dataset_cls = DatasetNTU
        smpl_model = 'smpl'
    elif cfg.dataset == 'humanact12':
        dataset_cls = DatasetACT12
        smpl_model = 'smpl'
    elif cfg.dataset == 'babel':
        dataset_cls = DatasetBabel
        smpl_model = 'smplh'

    # for act in cfg.vae_specs['actions']:
    dataset = dataset_cls('train', t_his, t_pred, actions='all', use_vel=cfg.use_vel,
                          acts=cfg.vae_specs['actions'] if 'actions' in cfg.vae_specs else None,
                          acts_class=cfg.vae_specs['action_classes'] if 'action_classes' in cfg.vae_specs else None,
                          max_len=cfg.vae_specs['max_len'] if 'max_len' in cfg.vae_specs else None,
                          min_len=cfg.vae_specs['min_len'] if 'min_len' in cfg.vae_specs else None,
                          is_6d=cfg.vae_specs['is_6d'] if 'is_6d' in cfg.vae_specs else False,
                          data_file=cfg.vae_specs['data_file'] if 'data_file' in cfg.vae_specs else None)

    """model"""
    if cfg.dataset == 'babel':
        dataset.traj_dim = 60
        model = get_action_vae_model(cfg, 60,max_len=dataset.max_len )#max_len=dataset.max_len - cfg.t_his + cfg.vae_specs['t_pre_extra']
    else:
        model = get_action_vae_model(cfg, dataset.traj_dim, max_len=dataset.max_len)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    if args.iter > 0:
        cp_path = cfg.vae_model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path) #"/home/llx/projects/human_pose/code/WAT/results/babel_rnn_p/models/vae_0500.p"
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])
    model.to(device)
    model.eval()


    val(args.iter)
