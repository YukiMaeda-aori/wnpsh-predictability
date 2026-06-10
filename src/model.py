import torch
import torch.nn as nn


'''
model definition
'''
class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_p=0.1, proj_p=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_p)
        self.to_out = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
        
    def forward(self, x):
        # x: (batch, seq_len, dim)
        x_in = x
        x = self.norm(x)
        
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, self.heads, -1).transpose(1,2), qkv)
        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1,2).reshape(b, n, -1)
        out = self.to_out(out)
        out = self.proj_drop(out)
        return x_in + out
        

class Conv3DAttention(nn.Module):
    def __init__(self, input_shape, attn_heads=8, attn_dim_head=64, dropout_p=0.2):
        super().__init__()
        in_ch, lag, H, W = input_shape
        
        # 3D Conv stack
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=32,
                      kernel_size=(2,2,2), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(2,2,2),
                      stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(2,2,2),
                      stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        
        # infer the output shape after the conv layers using a dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, lag, H, W)
            x = self.layer1(dummy)   # (1, C1, T1, H1, W1)
            x = self.layer2(x)       # (1, C2, T2, H2, W2)
            x = self.layer3(x)       # (1, C3, T3, H3, W3)
            _, C3, T3, H3, W3 = x.shape
            # Self-Attention operates on the sequence dimension
            seq_len = T3
            feat_dim = C3 * H3 * W3
            flattened_size = x.numel()
        
        
        self.proj_in_dim = 4096  # dimension of the projected feature space for attention
        self.proj_in = nn.Linear(feat_dim, self.proj_in_dim)
        self.attn1 = SelfAttention(dim=self.proj_in_dim, heads=attn_heads, dim_head=attn_dim_head)
        #self.attn2 = SelfAttention(dim=self.proj_in_dim, heads=attn_heads, dim_head=attn_dim_head)
        self.proj_out = nn.Linear(self.proj_in_dim, feat_dim)
        
        self.C3, self.T3, self.H3, self.W3 = C3, T3, H3, W3
        
        # branch1: mu, branch2: log_sigma
        self.fc_branch1 = nn.Sequential(
            nn.Linear(flattened_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.fc_branch2 = nn.Sequential(
            nn.Linear(flattened_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

        self.fc_mu = nn.Linear(32, 1)        
        self.fc_log_sigma = nn.Linear(32, 1)
    
    def forward(self, x):
        # x: (B, in_ch, lag, H, W)
        x = self.layer1(x)   # (B, C1, T1, H1, W1)
        x = self.layer2(x)   # (B, C2, T2, H2, W2)
        x = self.layer3(x)   # (B, C3, T3, H3, W3)
        
        B, C3, T3, H3, W3 = x.shape

        x_seq = x.permute(0, 2, 1, 3, 4).contiguous()     # (B, T3, C3, H3, W3)
        x_seq = x_seq.view(B, T3, C3 * H3 * W3)           # (B, T3, feat_dim)
        
        # linear -> Self-Attention -> linear
        x_seq = self.proj_in(x_seq)                       # (B, T3, proj_in_dim)
        x_seq = self.attn1(x_seq)                          # (B, T3, proj_in_dim)
        #x_seq = self.attn2(x_seq)                          # (B, T3, proj_in_dim)
        x_seq = self.proj_out(x_seq)                      # (B, T3, feat_dim)
        
        x_flat = x_seq.contiguous().view(B, -1)
        
        branch1 = self.fc_branch1(x_flat)
        branch2 = self.fc_branch2(x_flat)
        
        mu = self.fc_mu(branch1)
        log_sigma = self.fc_log_sigma(branch2)
        return mu, log_sigma