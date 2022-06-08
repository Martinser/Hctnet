# This code modify on the basis of ConvNeXt, details are quoted below:
# @Article{liu2022convnet,
#   author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
#   title   = {A ConvNet for the 2020s},
#   journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#   year    = {2022},
# }
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import torch.utils.model_zoo as model_zoo
from models.swin_transformer import SwinTransformerBlock as sw_block
from models.swin_transformer import BasicLayer as ba_block
from models.swin_transformer import PatchMerging
import json
from torchvision.models.resnet import Bottleneck
from models.crossformer import CrossFormerBlock as cf_block

class Block(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x

class Hctnet(nn.Module):
    """
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.lay4 = LayerNorm(384, eps=1e-6, data_format="channels_first")
        patch_size=[2,4]
        
        # edit by Ge Wu
        # Multiscale downsampling before stage3, follow CrossFormer(https://arxiv.org/pdf/2108.00154.pdf)
        self.reductions2 = nn.ModuleList()
        dim=384
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                out_dim = 2 * dim // 2 ** i
            else:
                out_dim = 2 * dim // 2 ** (i + 1)
            stride = 2
            padding = (ps - stride) // 2
            self.reductions2.append(nn.Conv2d(dim, out_dim, kernel_size=ps,stride=stride, padding=padding))


        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

        # edit by Ge Wu
        # The following is MSAs of Swin-trainsformer(https://arxiv.org/abs/2103.14030)
        depths = [2, 2, 6, 2]
        drop_path_rate=0.2
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        drop_path = dpr[sum(depths[:3]):sum(depths[:3 + 1])]
        self.sw1=sw_block(
            dim=768,
            input_resolution=(7,7),
            num_heads=24,
            window_size=7,
            shift_size=0,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=drop_path[0],
            norm_layer=nn.LayerNorm
        )
        self.sw2=sw_block(
            dim=768,
            input_resolution=(7,7),
            num_heads=24,
            window_size=7,
            shift_size=3,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=drop_path[1],
            norm_layer=nn.LayerNorm
        )
        self.norm2=nn.LayerNorm(768)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.softclass=nn.Linear(768,2)

        #The following is MSAs of CrossFormer(https://arxiv.org/pdf/2108.00154.pdf)
        self.cf1=cf_block(
            dim=768,
            input_resolution=(7, 7),
            num_heads=24,
            group_size=7,
            lsda_flag=0,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=drop_path[0],
            num_patch_size=2,
            norm_layer=nn.LayerNorm
        )
        self.cf2=cf_block(
            dim=768,
            input_resolution=(7, 7),
            num_heads=24,
            group_size=7,
            lsda_flag=1,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=drop_path[1],
            num_patch_size=2,
            norm_layer=nn.LayerNorm
        )

        #Resnet's Bottleneck(https://arxiv.org/abs/1512.03385)
        self.Bottleneck=Bottleneck(inplanes=768,planes=192,base_width=64)
        
        #Large Kernel Attention(LKA,https://arxiv.org/abs/2202.09741)
        self.conv0 = nn.Conv2d(768, 768, 5, padding=2, groups=768)
        self.conv_spatial = nn.Conv2d(768, 768, 7, stride=1, padding=9, groups=768, dilation=3)
        self.conv1 = nn.Conv2d(768, 768, 1)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            if i !=3:
                x = self.stages[i](x)

            '''
            Output of stage1~stage4:
            stage1        torch.Size([16, 96, 56, 56])
            stage2        torch.Size([16, 192, 28, 28])
            stage3        torch.Size([16, 384, 14, 14])
            stage4        torch.Size([16, 768, 7, 7])
            '''

        return x


    def forward(self, x):
        x=self.forward_features(x)

        #Add LKA
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        x=u * attn

        #Cross-combination of Convs and MSAs
        b, c, h, w = x.shape
        x=x.view(b,h,w,c)
        x=x.flatten(1,2)
        x=self.sw1(x)
        x=x.view(b,c,h,w)
        x=self.Bottleneck(x)
        x=x.view(b,h*w,c)
        x=self.sw2(x)

        x = self.norm2(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x=self.softclass(x)

        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
}

@register_model
def hctnet(pretrained=True, **kwargs):
    model = Hctnet(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        # Where the pre-weight is
        weights="/home/don/wg/ConvNeXt-main/weight/convnext_small_1k_224_ema.pth"
        weights_dict = torch.load(weights, map_location="cpu")["model"]
        # Remove weights about categorical categories
        for k in list(weights_dict.keys()):
            if "head" in k:
                print("Successfully deleted")
                del weights_dict[k]

        model.load_state_dict(weights_dict, strict=False)
    return model

