import os
import sys
import math
import pickle
import argparse
import time
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
sys.path.append(os.getcwd())
from utils import *
from motion_pred.utils.config import Config

from motion_pred.utils.dataset_ntu_act_transition import DatasetNTU
from motion_pred.utils.dataset_humanact12_act_transition import DatasetACT12
from motion_pred.utils.dataset_babel_action_transition import DatasetBabel
from models.motion_pred import *
from os.path import join as pjoin

def loss_function(label,label_est,h,criterion):

    loss = criterion(h,label)

    loss_r = loss
    with torch.no_grad():
        accuracy = (label == torch.where(label_est == label_est.max(dim=1,keepdims=True)[0])[-1]).sum()/float(len(label))
    # loss_r = MSE
    return loss_r, np.array([loss_r.item(),accuracy.item()])


def train(epoch):
    t_s = time.time()
    train_losses = 0
    total_num_sample = 0
    criterion = torch.nn.CrossEntropyLoss()
    loss_names = ['TOTAL', 'accuracy']
    sampler = WeightedRandomSampler(torch.DoubleTensor(dataset.sample_weight), int(dataset.data_len),replacement=True)
    # train_loader = DataLoader(dataset=dataset, # 要传递的数据集
    #                         batch_size=cfg.batch_size, #一个小批量数据的大小是多少
    #                         shuffle=True, # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
    #                         num_workers=0) # 需要几个进程来一次性读取这个小批量数据，默认0，一般用0就够了，多了有时会出一些底层错误。    for traj_np, 

    train_loader = DataLoader(dataset=dataset, # 要传递的数据集
                            batch_size=cfg.batch_size, #一个小批量数据的大小是多少
                            sampler = sampler,#shuffle=True, # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
                            num_workers=0) # 需要几个进程来一次性读取这个小批量数据，默认0，一般用0就够了，多了有时会出一些底层错误。    for traj_np, label, fn, fn_mask in generator:
        # traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
    for index,action_str,action_label,action_seq,action_len,fn_mask,fn_gt  in train_loader:
        if action_seq.device!=device:
            action_seq = action_seq.to(device)

        traj_tmp = action_seq.type(dtype).squeeze().contiguous()
        if action_label.device!=device:
            action_label = action_label.to(device)       
        action_label = action_label.type(dtype)
        # fn = tensor(fn, device=device, dtype=dtype)
        if fn_mask.device!=device:
            fn_mask = fn_mask.to(device)
        # print(fn_mask.shape,traj_tmp.shape)
        fn_mask = fn_mask.type(dtype).squeeze()
        # print(fn_mask.sum(dim=1))
        if fn_gt.device!=device:
            fn_gt = fn_gt.to(device)
        # print(fn_mask.shape,traj_tmp.shape)
        fn_gt = fn_gt.type(dtype).squeeze()   




        traj = action_seq #tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
        label = torch.where(action_label==1)[-1]
        fn = fn_gt#tensor(fn, device=device, dtype=dtype)
        fn_mask = tensor(fn_mask, device=device, dtype=dtype)
        X = traj_tmp.permute(1, 0, 2)
        if cfg.dataset == 'babel':
            index_used = list(range(30)) + list(range(36, 66))
            X = X[:, :, index_used]
        # print(X.shape)
        label_est,h = model(X,fn_gt)
        loss, losses = loss_function(label,label_est,h,criterion)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses += losses
        total_num_sample += 1

    scheduler.step()
    dt = time.time() - t_s
    train_losses /= total_num_sample
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    if(epoch%10==0):
        logger.info('====> Epoch: {} Time: {:.2f} {} lr: {:.5f}'.format(epoch, dt, losses_str, lr))
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalars(name, {'train':loss}, epoch)

def val(epoch):
    with torch.no_grad():
        t_s = time.time()
        train_losses = 0
        total_num_sample = 0
        criterion = torch.nn.CrossEntropyLoss()
        loss_names = ['accuracy']
        sampler = WeightedRandomSampler(torch.DoubleTensor(dataset_test.sample_weight), int(dataset_test.data_len),replacement=True)
        # test_loader = DataLoader(dataset=dataset_test, # 要传递的数据集
        #                     batch_size=cfg.batch_size, #一个小批量数据的大小是多少
        #                     shuffle=True, # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要

        #                     num_workers=0) # 需要几个进程来一次性读取这个小批量数据，默认0，一般用0就够了，多了有时会出一些底层错误。
        test_loader = DataLoader(dataset=dataset_test, # 要传递的数据集
                            batch_size=cfg.batch_size, #一个小批量数据的大小是多少
                            sampler=sampler,#                            shuffle=True, # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要

                            num_workers=0) # 需要几个进程来一次性读取这个小批量数据，默认0，一般用0就够了，多了有时会出一些底层错误。    for traj_np, label, fn, fn_mask in generator:
        # traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
        for index,action_str,action_label,action_seq,action_len,fn_mask,fn_gt  in test_loader:
            if action_seq.device!=device:
                action_seq = action_seq.to(device)

            traj_tmp = action_seq.type(dtype).squeeze().contiguous() 
            if action_label.device!=device:
                action_label = action_label.to(device)       
            action_label = action_label.type(dtype)
            # fn = tensor(fn, device=device, dtype=dtype)
            if fn_mask.device!=device:
                fn_mask = fn_mask.to(device)
            # print(fn_mask.shape,traj_tmp.shape)
            fn_mask = fn_mask.type(dtype).squeeze()
            # print(fn_mask.sum(dim=1))
            if fn_gt.device!=device:
                fn_gt = fn_gt.to(device)
            # print(fn_mask.shape,traj_tmp.shape)
            fn_gt = fn_gt.type(dtype).squeeze()   




            # traj = action_seq #tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
            label = torch.where(action_label==1)[-1]
            fn = fn_gt#tensor(fn, device=device, dtype=dtype)
            fn_mask = tensor(fn_mask, device=device, dtype=dtype)
            X = traj_tmp.permute(1, 0, 2)#
            if cfg.dataset == 'babel':
                index_used = list(range(30)) + list(range(36, 66))
                X = X[:, :, index_used]
            # print(X.shape,fn_gt.shape)
            # exit(0)
            # print(fn_gt.shape,X.shape)
            # print(fn_gt)
            label_est,h = model(X,fn_gt)
            loss, losses = loss_function(label,label_est,h,criterion)
            train_losses += losses[-1:]
            total_num_sample += 1

        dt = time.time() - t_s
        train_losses /= total_num_sample
        lr = optimizer.param_groups[0]['lr']
        global best_acc
        best_acc = max(best_acc,train_losses[0])

        losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
        losses_str = "best acc {:4f}".format(best_acc)+" " +losses_str
        if(epoch%10==0):
            logger.info('====> Val Epoch: {} Time: {:.2f} {} lr: {:.5f}'.format(epoch, dt, losses_str, lr))
        for name, loss in zip(loss_names, train_losses):
            tb_logger.add_scalars(name, {'val':loss}, epoch)
def get_mu_cov():
    with torch.no_grad():
        t_s = time.time()
        train_losses = 0
        total_num_sample = 0
        criterion = torch.nn.CrossEntropyLoss()
        loss_names = ['accuracy']
        sampler = WeightedRandomSampler(torch.DoubleTensor(dataset_test.sample_weight), int(dataset_test.data_len),replacement=True)
        test_loader = DataLoader(dataset=dataset_test, # 要传递的数据集
                            batch_size=cfg.batch_size, #一个小批量数据的大小是多少
                            sampler=sampler,#                            shuffle=True, # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要

                            num_workers=0) # 需要几个进程来一次性读取这个小批量数据，默认0，一般用0就够了，多了有时会出一些底层错误。    for traj_np, label, fn, fn_mask in generator:
        # test_loader = DataLoader(dataset=dataset_test, # 要传递的数据集
        #                     batch_size=cfg.batch_size, #一个小批量数据的大小是多少
        #                     shuffle=False, # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
        #                     num_workers=0) # 需要几个进程来一次性读取这个小批量数据，默认0，一般用0就够了，多了有时会出一些底层错误。    for traj_np, label, fn, fn_mask in generator:
        # # traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
        feat = []
        for index,action_str,action_label,action_seq,action_len,fn_mask,fn_gt  in test_loader:
            if action_seq.device!=device:
                action_seq = action_seq.to(device)
            traj_tmp = action_seq.type(dtype).squeeze().contiguous() 
            if action_label.device!=device:
                action_label = action_label.to(device)       
            action_label = action_label.type(dtype)
            # fn = tensor(fn, device=device, dtype=dtype)
            if fn_mask.device!=device:
                fn_mask = fn_mask.to(device)
            # print(fn_mask.shape,traj_tmp.shape)
            fn_mask = fn_mask.type(dtype).squeeze()
            # print(fn_mask.sum(dim=1))
            if fn_gt.device!=device:
                fn_gt = fn_gt.to(device)
            # print(fn_mask.shape,traj_tmp.shape)
            fn_gt = fn_gt.type(dtype).squeeze()   




            # traj = action_seq #tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
            label = torch.where(action_label==1)[-1]
            fn = fn_gt#tensor(fn, device=device, dtype=dtype)
            fn_mask = tensor(fn_mask, device=device, dtype=dtype)
            X = traj_tmp.permute(1, 0, 2)#
            if cfg.dataset == 'babel':
                index_used = list(range(30)) + list(range(36, 66))
                X = X[:, :, index_used]
            # print(X.shape,fn_gt.shape)
            # exit(0)
            # print(fn_gt.shape,X.shape)
            # print(fn_gt)
            label_est,h,h_x = model(X,fn_gt,is_feat=True)
            # loss, losses = loss_function(label,label_est,h,criterion)
            # train_losses += losses[-1:]
            # total_num_sample += 1
            feat.append(h_x.cpu().data.numpy())
        feat = np.concatenate(feat, axis=0)
        mu_test = feat.mean(axis=0)
        cov_test = np.matmul(feat.transpose(1, 0), feat) / feat.shape[0]

        sampler = WeightedRandomSampler(torch.DoubleTensor(dataset.sample_weight), int(dataset.data_len),replacement=True)
        train_loader = DataLoader(dataset=dataset, # 要传递的数据集
                            batch_size=cfg.batch_size, #一个小批量数据的大小是多少
                            sampler=sampler, # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
                            num_workers=0) 
        # train_loader = DataLoader(dataset=dataset, # 要传递的数据集
        #                     batch_size=cfg.batch_size, #一个小批量数据的大小是多少
        #                     shuffle=False, # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
        #                     num_workers=0) 
        feat = []

        for index,action_str,action_label,action_seq,action_len,fn_mask,fn_gt  in train_loader:
            if action_seq.device!=device:
                action_seq = action_seq.to(device)
            traj_tmp = action_seq.type(dtype).squeeze().contiguous() 
            if action_label.device!=device:
                action_label = action_label.to(device)       
            action_label = action_label.type(dtype)
            # fn = tensor(fn, device=device, dtype=dtype)
            if fn_mask.device!=device:
                fn_mask = fn_mask.to(device)
            # print(fn_mask.shape,traj_tmp.shape)
            fn_mask = fn_mask.type(dtype).squeeze()
            # print(fn_mask.sum(dim=1))
            if fn_gt.device!=device:
                fn_gt = fn_gt.to(device)
            # print(fn_mask.shape,traj_tmp.shape)
            fn_gt = fn_gt.type(dtype).squeeze()   




            # traj = action_seq #tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
            label = torch.where(action_label==1)[-1]
            fn = fn_gt#tensor(fn, device=device, dtype=dtype)
            fn_mask = tensor(fn_mask, device=device, dtype=dtype)
            X = traj_tmp.permute(1, 0, 2)#
            if cfg.dataset == 'babel':
                index_used = list(range(30)) + list(range(36, 66))
                X = X[:, :, index_used]
            # print(X.shape,fn_gt.shape)
            # exit(0)
            # print(fn_gt.shape,X.shape)
            # print(fn_gt)
            label_est,h,h_x = model(X,fn_gt,is_feat=True)
            # loss, losses = loss_function(label,label_est,h,criterion)
            # train_losses += losses[-1:]
            # total_num_sample += 1
            feat.append(h_x.cpu().data.numpy())
        feat = np.concatenate(feat, axis=0)
        mu_train = feat.mean(axis=0)
        cov_train = np.matmul(feat.transpose(1, 0), feat) / feat.shape[0]
        return mu_test,cov_test,mu_train,cov_train
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='babel_act_classifier_nact5')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test', action='store_false', default=False)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    cfg = Config(args.cfg, test=args.test)
    tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

    """parameter"""
    mode = args.mode
    nz = cfg.nz
    t_his = cfg.t_his
    t_pred = cfg.t_pred

    """data"""

    dataset_cls = DatasetBabel

    dataset = dataset_cls(args.mode, t_his, t_pred, actions='all', use_vel=cfg.use_vel,
                          acts=cfg.vae_specs['actions'] if 'actions' in cfg.vae_specs else None,
                          max_len=cfg.vae_specs['max_len'] if 'max_len' in cfg.vae_specs else None,
                          min_len=cfg.vae_specs['min_len'] if 'min_len' in cfg.vae_specs else None,
                          is_6d=cfg.vae_specs['is_6d'] if 'is_6d' in cfg.vae_specs else False,
                          data_file=cfg.vae_specs['data_file'] if 'data_file' in cfg.vae_specs else None)
    dataset_test = dataset_cls('test', t_his, t_pred, actions='all', use_vel=cfg.use_vel,
                               acts=cfg.vae_specs['actions'] if 'actions' in cfg.vae_specs else None,
                               max_len=cfg.vae_specs['max_len'] if 'max_len' in cfg.vae_specs else None,
                               min_len=cfg.vae_specs['min_len'] if 'min_len' in cfg.vae_specs else None,
                               is_6d=cfg.vae_specs['is_6d'] if 'is_6d' in cfg.vae_specs else False,
                               data_file=cfg.vae_specs['data_file'] if 'data_file' in cfg.vae_specs else None)

    if cfg.normalize_data:
            dataset.normalize_data()

    """model"""
    model = get_action_classifier(cfg, 60, max_len=dataset.max_len)
    optimizer = optim.Adam(model.parameters(), lr=cfg.vae_lr)
    scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.num_vae_epoch_fix, nepoch=cfg.num_vae_epoch)
    logger.info(">>> total params: {:.2f}M".format(sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    if args.iter > 0:
        cp_path = cfg.vae_model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])
    global best_acc
    best_acc = 0
    if mode == 'train':
        model.to(device)
        model.train()
        for i in range(args.iter, cfg.num_vae_epoch):
            train(i)
            val(i)
            if cfg.save_model_interval > 0 and (i + 1) % cfg.save_model_interval == 0:
                with to_cpu(model):
                    cp_path = cfg.vae_model_path % (i + 1)
                    model_cp = {'model_dict': model.state_dict(), 'meta': {'std': dataset.std, 'mean': dataset.mean}}
                    pickle.dump(model_cp, open(cp_path, 'wb'))
    mu_test,cov_test,mu_train,cov_train = get_mu_cov()
    save_name = pjoin(cfg.result_dir,"mu_cov.npz")
    # print({"mu_test":mu_test,"cov_test":cov_test,"mu_train":mu_train,"cov_train":cov_train})
    np.savez(save_name,data={"mu_test":mu_test,"cov_test":cov_test,"mu_train":mu_train,"cov_train":cov_train})
    
