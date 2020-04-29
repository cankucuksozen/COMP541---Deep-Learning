import torch
import torchvision

from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

class selfAttn2d(nn.Module):
    
    def __init__(self, input_dims, kernel_size, stride, padding, Nh, dk, dv):
        super(selfAttn2d, self).__init__()
        
        self.input_dim = input_dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.dkh = self.dk // self.Nh
        self.dvh = self.dv // self.Nh
        
        self.unfold = nn.Unfold(kernel_size = kernel_size, stride = stride, padding = padding)
        
        self.conv_q = nn.Conv2d(input_dims, dk, 1, stride = 1, padding = 0, bias = False)
        self.conv_k = nn.Conv2d(input_dims, dk, 1, stride = 1, padding = 0, bias = False)
        self.conv_v = nn.Conv2d(input_dims, dv, 1, stride = 1, padding = 0, bias = False)
        
        self.linear_sf1 = nn.Linear(2*self.dkh, self.dvh)
        self.tanh = nn.Tanh()
        self.linear_sf2 = nn.Linear(self.dvh, kernel_size ** 2)
        
        self.conv_rel = nn.Conv2d(input_dims, dk, 3, stride = 1, padding = 1, bias = False)
        self.deconv_rel = nn.ConvTranspose2d(dk, dk, self.kernel_size, stride = self.kernel_size, 
                                             bias = False)
        
        #self.conv_attn = nn.Conv2d(dv, dv, 1, stride = 1, padding = 0, bias = False)
        self.attn_proj = nn.Linear(dv, dv)
            
        for i in [self.conv_q, self.conv_k, self.conv_v, self.conv_rel, self.deconv_rel]:
            nn.init.kaiming_uniform_(i.weight, a=1)
    
        self.softmax = nn.Softmax(dim=3)
        
    def forward(self, x):
        
        b, c, h, w = x.shape
        odims = self.odims(x)
        
        patches = self.unfold(x)
        _, _, num_q = patches.shape
        patches = patches.permute(0,2,1)
        spatial_patches = torch.reshape(patches, (b*num_q, c, self.kernel_size ,self.kernel_size))
        
        q = self.conv_q(spatial_patches)
        k = self.conv_k(spatial_patches)
        v = self.conv_v(spatial_patches)
        
        rel = self.conv_rel(spatial_patches)
        rel = self.deconv_rel(rel)
        
        q *= self.dkh ** -0.5

        q = self.split_heads_2d(q, self.Nh)
        k = self.split_heads_2d(k, self.Nh)
        v = self.split_heads_2d(v, self.Nh)
        rel = self.split_heads_2d(rel, self.Nh)
        
        flat_q = self.flatten_hw(q)
        flat_k = self.flatten_hw(k)
        flat_v = self.flatten_hw(v)
        
        logits = self.attn_score_function(flat_q, flat_k)
        rel_logits = self.rel_logits_2d(q, rel)
        logits += rel_logits
        
        weights = self.softmax(logits)
        flat_v =  flat_v.permute(0,1,3,2)
        
        attn = torch.matmul(weights, flat_v)
        attn = self.combine_heads_2d(attn)
        
        attn = self.attn_proj(attn).permute(0,2,1)
        attn_out = torch.sum(attn, dim=2)    
        
        attn_out = torch.reshape(attn_out, (b, num_q, -1)).permute(0,2,1)       
        attn_out = torch.reshape(attn_out, odims)
        
        return attn_out
        
    def odims(self, x):
        b, _, h, w = x.shape
        odims_h = int((h-self.kernel_size+2*self.padding)//self.stride)+1
        odims = (b, self.dv, odims_h, odims_h)
        return odims    
        
    def split_heads_2d(self, x, Nh):
        b, d, h, w = x.shape
        ret_shape = (b, self.Nh, d // self.Nh, h, w)
        out = torch.reshape(x, ret_shape)
        return out
    
    def flatten_hw(self, x):
        if x.dim() == 5:
            b, Nh, d, h, w = x.shape
            new_size = (b, Nh, d, h*w)
            out = torch.reshape(x, new_size)
        """
        else:
            b, d, h, w = x.shape
            new_size = (b, d, h*w)
            out = torch.reshape(x, new_size).permute(0,2,1)
        """
        return out 
    
    def attn_score_function(self, q, k):
        x = torch.cat((q,k), dim = 2)
        x = x.permute(0,1,3,2)
        x = self.linear_sf1(x)
        x = self.tanh(x)
        x = self.linear_sf2(x)
        return x
    
    
    def flatten_rel(self, x):
        b, Nh, d, k2, k2 = x.shape
        new_size = (b, Nh, d, self.kernel_size, self.kernel_size, self.kernel_size, 
                    self.kernel_size)
        out = torch.reshape(x, new_size)
        out = out.permute(0,1,2,3,5,4,6)
        return out
    
    def rel_logits_2d(self, q, rel):
        b, Nh, d, h, w = q.shape
        rel = self.flatten_rel(rel)
        q = torch.reshape(q, (b, Nh, d, h, w, 1, 1))
        q = q.expand_as(rel)
        out = rel * q
        rel_logits = torch.sum(out, dim = 2)
        rel_logits = torch.reshape(rel_logits, (b, Nh, h*w, h*w))
        return rel_logits
    
    def combine_heads_2d(self, x):
        b, Nh, h2, d = x.shape
        ret_shape = (b, h2, Nh*d)
        out = torch.reshape(x, ret_shape)
        return out
