import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.distributions.normal import Normal
from callbacks import EarlyStopping
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from model import Conv3DAttention


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def indexing(lead_time, real_time, sup_data): 
    rt = real_time[4:-lead_time-1]
    t_data = sup_data[lead_time+4:-1]

    idx = np.where((rt.year <= 2015))[0]
    #idx = np.where((rt.year <= 2015)& (rt.month >= 5) & (rt.month <= 9))[0]
    #idx = np.where(((rt.year <= 2007) | (rt.year > 2015)) & (rt.month >= 5) & (rt.month <= 10))[0]
    #idx = np.where(((rt.year <= 1999) | (rt.year > 2007)) & (rt.month >= 5) & (rt.month <= 9))[0]
    t_train = t_data[idx]
    idx = np.where((rt.year > 2015))[0]
    #idx = np.where((rt.year > 2015) & (rt.month >= 5) & (rt.month <= 9))[0]
    #idx = np.where((rt.year > 2007) & (rt.year <= 2015) & (rt.month >= 5) & (rt.month <= 10))[0]
    #idx = np.where((rt.year > 1999) & (rt.year <= 2007) & (rt.month >= 5) & (rt.month <= 9))[0]
    t_test = t_data[idx]
    rt_test = rt[idx]
    print('t_train, t_test =', t_train.shape, t_test.shape)
    return rt, rt_test, t_train, t_test

def preprocess(data, real_time):
    rt = real_time[4:-1]
    ipt = np.stack([data[4-lag:-1-lag] for lag in range(5)], axis=1)
    idx1 = np.where((rt.year <= 2015))[0]
    #idx1 = np.where((rt.year <= 2015) & (rt.month >= 5) & (rt.month <= 9))[0]
    #idx1 = np.where(((rt.year <= 2007) | (rt.year > 2015)) & (rt.month >= 5) & (rt.month <= 10))[0]
    #idx1 = np.where(((rt.year <= 1999) | (rt.year > 2007)) & (rt.month >= 5) & (rt.month <= 9))[0]
    ipt_train = ipt[idx1]
    idx2 = np.where((rt.year > 2015))[0]
    #idx2 = np.where((rt.year > 2015) & (rt.month >= 5) & (rt.month <= 9))[0]
    #idx2 = np.where((rt.year > 2007) & (rt.year <= 2015) & (rt.month >= 5) & (rt.month <= 10))[0]
    #idx2 = np.where((rt.year > 1999) & (rt.year <= 2007) & (rt.month >= 5) & (rt.month <= 9))[0]
    ipt_test = ipt[idx2]
    return ipt_train, ipt_test



class CRPSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sqrt_pi_inv = 1.0 / torch.sqrt(torch.tensor(np.pi))
        
    def forward(self, y, mu, log_sigma):
        sigma = torch.exp(log_sigma) + 1e-6
        z = (y - mu) / sigma
        normal = Normal(0, 1)
        cdf = normal.cdf(z)
        pdf = torch.exp(normal.log_prob(z))
        crps = sigma * (z * (2 * cdf - 1) + 2 * pdf - self.sqrt_pi_inv)
        return crps.mean()

        
def culc_cor(predict, y_test, lead_time):
    cor = np.corrcoef(predict[:], y_test[:,0])[0,1]
    return cor
  

'''
main program
'''
if __name__ == '__main__':
    #np.random.seed(123)
    #torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    C = 12
    Ttr = 5661
    Tte = 1224
    x_train_t = torch.empty((Ttr, C, 5, 30, 144), dtype=torch.float32)
    x_test_t  = torch.empty((Tte, C, 5, 30, 144), dtype=torch.float32)
    
    path = './data/era5/ipt_data'
    variables = ['mtnlwrf', 'z850', 'z500', 'z200', 'u850', 'u200', 'v850', 'v200', 'w500', 't2m', 'tcwv']
                 
    for c, variable in enumerate(variables):
        data = np.load(f'{path}/{variable}.npz')
        #print('Loading:', member, variable)
        if variable == 'mtnlwrf':
          data_i = - data['data'][:,13:43,:]
        else:
          data_i = data['data'][:,13:43,:]
        real_time = pd.to_datetime(data['rt'][:])
        x_train_i, x_test_i = preprocess(data_i, real_time)
        x_train_i = x_train_i.astype(np.float32, copy=False)
        x_test_i  = x_test_i.astype(np.float32, copy=False)
        std = x_train_i.std(axis=0, dtype=np.float32) + 1e-5
        x_train_norm = x_train_i / std
        x_test_norm = x_test_i / std
        #print(x_train_norm.shape[0], x_test_norm.shape[0])
        x_train_t[:, c] = torch.from_numpy(x_train_norm)
        x_test_t[:, c] = torch.from_numpy(x_test_norm)

    print('x_train_t, x_test_t =', x_train_t.shape, x_test_t.shape)
    time = data['rt'][:]
    real_time = pd.to_datetime(time)
    data_file = './data/era5/idx/wnpsh_eddy_index.npz'
    wpsh     = np.load(data_file)['wnpsh'][:,np.newaxis,0]

    def running_mean(data, window):
        df = pd.DataFrame(data.reshape(data.shape[0], -1))
        rm = df.rolling(window, min_periods=1).mean().values    # min_periods=1: fill with NaN if data is missing
        return rm.reshape(data.shape)
   
    #wpsh_run = running_mean(wpsh, 5)
    #print("norm = ", wpsh_run.std())
    #wpsh_n = wpsh_run / 1736.15
  
    '''
    training description
    '''
    def train_step(x, t):
        model.train()
        optimizer.zero_grad()
        mu, log_sigma = model(x)  # malti task model の場合
        crps = loss_fn(t, mu, log_sigma)
        preds = torch.cat((mu, log_sigma), dim=1)
        crps.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return crps, preds
    
    def test_step(x, t):
        model.eval()
        mu, log_sigma = model(x) 
        crps = loss_fn(t, mu, log_sigma)
        preds = torch.cat((mu, log_sigma), dim=1)
        return crps, preds
  
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 1.0
        
    def cosine_annealing_lr(epoch, T_max, eta_max, eta_min):
        lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * epoch / T_max))
        return lr

    def warmup_cosine_lambda(epoch):
        if epoch < warmup_epochs:
        # warmup
            return epoch / warmup_epochs
        # cosine annealing
        progress = (epoch - warmup_epochs) / (50 - warmup_epochs)
        return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * progress))

    
    #lt_box = [8]
    lt_box = np.arange(0, 31)
  
    for lead_time in lt_box:
        print('==== lead time : {} day ====='.format(lead_time))
        # answer data
        rt, rt_test, t_train, t_test = indexing(lead_time, real_time, sup_data=wpsh)
        print('rt, t_train, t_test = ', rt.shape, t_train.shape, t_test.shape)
        
        #batch_size = 128
        #batch_size = 256
        batch_size = 64
        epoch_num = 50
        warmup_epochs = 5
        eta_min = 1e-5
        eta_max =1.0
        
        n_batches = x_train_t.shape[0] // batch_size
        n_test_batches = x_test_t.shape[0] // batch_size
    
        train_dataset = torch.utils.data.TensorDataset(
            x_train_t, 
            torch.Tensor(t_train),
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True)
        
        t_test_loader = t_test.copy()
        test_dataset = torch.utils.data.TensorDataset(
            x_test_t, 
            torch.Tensor(t_test_loader), 
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True)
    
        for seed in range(10):
          print('Seed = ', seed)
          model_path = f'./results/model/pre-training/{lead_time:02}/{seed:03}_m010.pth'
          state_dict = torch.load(model_path)
          model = Conv3DAttention(input_shape=x_train_t[0].shape)
          model.load_state_dict(state_dict)

          model = nn.DataParallel(model)
          model = model.to(device)
          es = EarlyStopping(patience=5, 
                             verbose=0,   # EalyStopping Counter (0/1)
                             path=f'./results/model/fine-tuning/{lead_time:02}/{seed:03}_m010_2016-2023.pth'
                             )
          loss_fn = CRPSLoss()
          optimizer = optimizers.AdamW(model.parameters(), weight_decay=0.01, lr=1e-6)
            
          scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)
            
          for epoch in range(epoch_num):
            train_loss = 0.
            test_loss = 0.
            loss = 0.
            preds_list = []
    
            for x_, t_ in train_loader:
                x_ = x_.to(device)
                t_ = t_.to(device)
                loss, _ = train_step(x_, t_) 
                train_loss += loss.item()
     
            scheduler.step()
              
            with torch.no_grad():
                for x_, t_ in test_loader:
                    x_ = torch.Tensor(x_).to(device)
                    t_ = torch.Tensor(t_).to(device)
                    tl_, pt_ = test_step(x_, t_)
                    test_loss += tl_
                    preds_list.append(pt_)

            preds_test = torch.cat(preds_list, dim=0)
            print('epoch: {}, loss: {:.3}, test loss: {:.3}, lr: {:.3}'.format(
              epoch+1,
              train_loss/n_batches,
              test_loss/n_test_batches,
              optimizer.param_groups[0]['lr'],
              ))

            es_counter = es(test_loss.item(), model)
            if (epoch > warmup_epochs) & (es_counter >= 5):
              print('Early stopping')
              break
            elif es_counter == 0:
              best_pred = preds_test
              
          predict = preds_test.cpu().detach().numpy()
          print(predict.shape) 
          
          cor1 = culc_cor(predict[:,0], t_test, lead_time)

          print('lead time {} day: WNPSH  = '.format(lead_time), cor1)

          
          np.savez(f"./results/array/CRPS-Conv3DSA/fine-tuning/{lead_time:02}/{seed:03}_m010_2016-2023.npz", predict=predict, wpsh=t_test)
                    
print('==== Finish! ====')

