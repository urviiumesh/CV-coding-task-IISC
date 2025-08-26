



import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SwinIRBlock']

def window_partition(x, window_size):

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return x

def window_reverse(windows, window_size, H, W):
    num_windows_total = windows.shape[0]
    windows_per_image = (H // window_size) * (W // window_size)
    B = num_windows_total // windows_per_image
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
    
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  

        if mask is not None:
            num_windows = mask.shape[0]
            assert B_ % num_windows == 0
            batch_repeat = B_ // num_windows
            mask_expanded = mask.repeat(batch_repeat, 1, 1)  
            attn = attn + mask_expanded.unsqueeze(1)  

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)  
        out = out.transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class SwinIRBlock(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dim=32,
                 input_resolution=(32, 32),
                 num_heads=4,
                 window_size=8,
                 shift_size=None,
                 mlp_ratio=4.0,
                 qkv_bias=True):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        H, W = input_resolution
        assert H % window_size == 0 and W % window_size == 0, "H and W must be divisible by window_size"

        self.window_size = window_size
        self.shift_size = (window_size // 2) if (shift_size is None and window_size > 1) else (shift_size or 0)
        assert 0 <= self.shift_size < self.window_size

        
        self.conv_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1)
        self.conv_unembed = nn.Conv2d(embed_dim, in_chans, kernel_size=3, padding=1)

        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WindowAttention(embed_dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim, hidden_features=int(embed_dim * mlp_ratio))

        
        self.register_buffer("attn_mask", None, persistent=False)

    def _create_attn_mask(self, H, W, device):
        if self.shift_size == 0:
            return None

        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        B, C_in, H, W = x.shape
        assert (H, W) == self.input_resolution, f"expected input resolution {self.input_resolution}, got {(H, W)}"

        device = x.device
        if (self.attn_mask is None) or (self.attn_mask.shape[1] != self.window_size * self.window_size):
            self.attn_mask = self._create_attn_mask(H, W, device)

        
        x_emb = self.conv_embed(x)  
        x_perm = x_emb.permute(0, 2, 3, 1).contiguous()  

        
        if self.shift_size > 0:
            shifted_x = torch.roll(x_perm, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_perm

        
        x_windows = window_partition(shifted_x, self.window_size)  
        x_windows = x_windows.view(-1, self.window_size * self.window_size, self.embed_dim)  

        
        x_windows_norm = self.norm1(x_windows)
        attn_windows = self.attn(x_windows_norm, mask=self.attn_mask)  
        x_windows = x_windows + attn_windows

        
        x_windows = x_windows.view(-1, self.window_size, self.window_size, self.embed_dim)
        shifted_x = window_reverse(x_windows, self.window_size, H, W)  

        
        if self.shift_size > 0:
            x_out = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_out = shifted_x

        x_out = x_out.permute(0, 3, 1, 2).contiguous()  

        
        x_mlp_prep = x_out.permute(0, 2, 3, 1).contiguous()  
        BHW = B * H * W
        x_mlp_flat = x_mlp_prep.view(BHW, self.embed_dim)
        x_mlp_norm = self.norm2(x_mlp_flat)
        x_mlp = self.mlp(x_mlp_norm)
        x_mlp = x_mlp.view(B, H, W, self.embed_dim).permute(0, 3, 1, 2).contiguous()  

        x_final = x_out + x_mlp

        
        x_recon = self.conv_unembed(x_final)  
        out = x + x_recon
        return out
