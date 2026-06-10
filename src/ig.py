import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients 
from model import Conv3DAttention


def indexing(lead_time, real_time, sup_data): 
    rt = real_time[4:-lead_time-1]
    t_data = sup_data[lead_time+4:-1]
    print(t_data.shape)
    #idx = np.where((rt.year <= 2015))[0]
    #idx = np.where((rt.year <= 2015)& (rt.month >= 6) & (rt.month <= 9))[0]
    #idx = np.where(((rt.year <= 2007) | (rt.year > 2015)) & (rt.month >= 6) & (rt.month <= 9))[0]
    idx = np.where(((rt.year <= 1999) | (rt.year > 2007)) & (rt.month >= 6) & (rt.month <= 9))[0]
    t_train = t_data[idx]
    #idx = np.where((rt.year > 2015))[0]
    #idx = np.where((rt.year > 2015) & (rt.month >= 6) & (rt.month <= 9))[0]
    #idx = np.where((rt.year > 2007) & (rt.year <= 2015) & (rt.month >= 6) & (rt.month <= 9))[0]
    idx = np.where((rt.year > 1999) & (rt.year <= 2007) & (rt.month >= 6) & (rt.month <= 9))[0]
    t_test = t_data[idx]
    rt_test = rt[idx]
    print('t_train, t_test =', t_train.shape, t_test.shape)
    return rt, rt_test, t_train, t_test


def preprocess(data, real_time):
    rt = real_time[4:-1]
    ipt = np.stack([data[4-lag:-1-lag] for lag in range(5)], axis=1)
    print(ipt.shape)
    #idx1 = np.where((rt.year <= 2015))[0]
    #idx1 = np.where((rt.year <= 2015) & (rt.month >= 6) & (rt.month <= 9))[0]
    #idx1 = np.where(((rt.year <= 2007) | (rt.year > 2015)) & (rt.month >= 6) & (rt.month <= 9))[0]
    idx1 = np.where(((rt.year <= 1999) | (rt.year > 2007)) & (rt.month >= 6) & (rt.month <= 9))[0]
    ipt_train = ipt[idx1]
    #idx2 = np.where((rt.year > 2015))[0]
    #idx2 = np.where((rt.year > 2015) & (rt.month >= 6) & (rt.month <= 9))[0]
    #idx2 = np.where((rt.year > 2007) & (rt.year <= 2015) & (rt.month >= 6) & (rt.month <= 9))[0]
    idx2 = np.where((rt.year > 1999) & (rt.year <= 2007) & (rt.month >= 6) & (rt.month <= 9))[0]
    ipt_test = ipt[idx2]
    return ipt_train, ipt_test


class MuLogSigmaWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        mu, log_sigma = self.base_model(x)   #  (B,1) 
        if mu.dim() == 1:
            mu = mu.unsqueeze(1)
        if log_sigma.dim() == 1:
            log_sigma = log_sigma.unsqueeze(1)
        return torch.cat([mu, log_sigma], dim=1)  # (B,2)


def ig_attribution(wrapped, inputs, baseline, out_idx, n_steps=64, internal_bs=2):
    """
    wrapped: MuLogSigmaWrapper(model) -> (B,2) 
    out_idx: 0 -> mu, 1 -> log_sigma
    """
    def forward_func(x):
        return wrapped(x)[:, out_idx]  # (B,)

    ig = IntegratedGradients(forward_func)
    inputs = inputs.requires_grad_(True)

    attr, delta = ig.attribute(
        inputs=inputs,
        baselines=baseline,            
        n_steps=n_steps,
        internal_batch_size=internal_bs,
        return_convergence_delta=True
    )
    return attr.detach().cpu().numpy(), delta.detach().cpu().numpy()


"""
main program
"""

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    C = 12
    Tte = 976
    x_test_t  = torch.empty((Tte, C, 5, 30, 144), dtype=torch.float32)
    
    path = './data/era5/ipt_data'
    variables = ['mtnlwrf', 'z850', 'z500', 'z200', 'u850', 'u200', 'v850', 'v200', 'w500', 't2m', 'tcwv']

                 
    for c, variable in enumerate(variables):
        data = np.load(f'{path}/{variable}.npz')
        
        if variable == 'mtnlwrf':
            data_i = - data['data'][:,13:43,:]  
        else:
            data_i = data['data'][:,13:43,:]
        real_time = pd.to_datetime(data['rt'][:])
        x_train_i, x_test_i = preprocess(data_i, real_time)
        x_train_i = x_train_i.astype(np.float32, copy=False)
        x_test_i  = x_test_i.astype(np.float32, copy=False)
        std = x_train_i.std(axis=0, dtype=np.float32) + 1e-5
        x_test_norm = x_test_i / std
        x_test_t[:, c] = torch.from_numpy(x_test_norm)

    print('x_test_t =', x_test_t.shape)  
    time = data['rt'][:]
    real_time = pd.to_datetime(time)
    
    # datasets: torch.Tensor, shape (N, C, lag, H, W), JJAS only, normalized
    datasets = torch.tensor(x_test_t, dtype=torch.float32).to(device)

    # Baseline is the climatology of JJAS (spatial mean of the normalized data)
    baseline = datasets.mean(dim=0, keepdim=True)  # (1,C,lag,H,W)

    lt_box = np.arange(0,31,3)
    #lt_box = [9]
    for lt in lt_box:
        attr_mu_sum = None
        attr_ls_sum = None
        delta_mu_sum = None
        delta_ls_sum = None
        for seed in range(10):
            print("seed = ", seed)
            model_path = f'./results/model/fine-tuning/{lt:02}/{seed:03}_m010_2016-2023.pth'
            model = Conv3DAttention(input_shape=datasets[0].shape).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            wrapped = MuLogSigmaWrapper(model).to(device).eval()

            x = datasets[:]

            attr_mu, delta_mu = ig_attribution(wrapped, x, baseline, out_idx=0)
            attr_ls, delta_ls = ig_attribution(wrapped, x, baseline, out_idx=1)

            if attr_mu_sum is None:
                attr_mu_sum = np.zeros_like(attr_mu, dtype=np.float32)
                attr_ls_sum = np.zeros_like(attr_ls, dtype=np.float32)
                delta_mu_sum = np.zeros_like(delta_mu, dtype=np.float32)
                delta_ls_sum = np.zeros_like(delta_ls, dtype=np.float32)

            np.add(attr_mu_sum, attr_mu.astype(np.float32, copy=False), out=attr_mu_sum)
            np.add(attr_ls_sum, attr_ls.astype(np.float32, copy=False), out=attr_ls_sum)
            np.add(delta_mu_sum, delta_mu.astype(np.float32, copy=False), out=delta_mu_sum)
            np.add(delta_ls_sum, delta_ls.astype(np.float32, copy=False), out=delta_ls_sum)
            del attr_mu, attr_ls, delta_mu, delta_ls

        # seed mean
        inv = np.float32(1.0 / 10.0)
        attr_mu_mean = attr_mu_sum * inv
        attr_ls_mean = attr_ls_sum * inv
        delta_mu_mean = delta_mu_sum * inv
        delta_ls_mean = delta_ls_sum * inv
    
        np.savez(
            f"./results/ig/crps_m10/{lt:02}/seed_mean_2016-2023.npz",
            attr_mu=attr_mu_mean,
            attr_log_sigma=attr_ls_mean,
            delta_mu=delta_mu_mean,
            delta_log_sigma=delta_ls_mean,
            time=time[:],
        )
        print("saved:", lt, seed, attr_mu_mean.shape, attr_ls_mean.shape, delta_mu_mean.mean(), delta_ls_mean.mean())
