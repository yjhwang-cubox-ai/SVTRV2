from typing import Dict, List, Optional, Tuple, Union, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OverlapPatchEmbed(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 embed_dims: int = 768,
                 num_layers: int = 2):
        
        super().__init__()        
        assert num_layers in [2, 3], \
            'The number of layers must belong to [2,3]'

        self.net = nn.Sequential()
        for num in range(num_layers, 0 ,-1):
            if (num == num_layers):
                _input = in_channels
            _output = embed_dims // (2**(num-1))
            self.net.add_module(
                f'ConvModule{str(num_layers - num)}',
                nn.Conv2d(
                    in_channels=_input,
                    out_channels=_output,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False #BatchNorm 을 사용할 때는 bias = False 가 일반적
                ))
            self.net.add_module(
                f'bn{str(num_layers - num)}',
                nn.BatchNorm2d(_output)
            )
            self.net.add_module(
                f'gelu{str(num_layers - num)}',
                nn.GELU()
            )
            _input = _output
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (Tensor): A Tensor of shape :math:`(N, C, H, W)`.

        Returns:
            Tensor: A tensor of shape math:`(N, HW//16, C)`.
        """
        x = self.net(x).flatten(2).permute(0, 2, 1)
        return x

class ConvMixer(nn.Module):
    """The conv Mixer.

    Args:
        embed_dims (int): Number of character components.
        num_heads (int, optional): Number of heads. Defaults to 8.
        input_shape (Tuple[int, int], optional): The shape of input [H, W].
            Defaults to [8, 25].
        local_k (Tuple[int, int], optional): Window size. Defaults to [3, 3].
    """
    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 input_shape: Tuple[int, int] = (8, 25),
                 local_k: Tuple[int, int] = (3,3)):
        super().__init__()

        self.input_shape = input_shape
        self.embed_dims = embed_dims
        self.local_mixer = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=local_k,
            stride=1,
            padding=(local_k[0] // 2, local_k[1] // 2),
            groups=num_heads
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, HW, C)`.

        Returns:
            torch.Tensor: A tensor of shape math:`(N, HW, C)`.
        """
        h, w = self.input_shape
        # reshape to 4D tensor for convolution
        x = x.permute(0, 2, 1).reshape(-1, self.embed_dims, h, w)
        # apply local mixing
        x = self.local_mixer(x)
        # reshape back to sequence form
        x = x.flatten(2).permute(0, 2, 1)
        return x

class AttnMixer(nn.Module):
    """One of mixer of {'Global', 'Local'}. Defaults to Global Mixer.

    Args:
        embed_dims (int): Number of character components.
        num_heads (int): Number of heads. Defaults to 8.
        mixer (str): The mixer type, choices are 'Global' and 'Local'.
            Defaults to 'Global'.
        input_shape (Tuple[int, int]): The shape of input [H, W].
            Defaults to (8, 25).
        local_k (Tuple[int, int]): Window size. Defaults to (7, 11).
        qkv_bias (bool): Whether a additive bias is required.
            Defaults to False.
        qk_scale (float): A scaling factor. Defaults to None.
        attn_drop (float): Attn dropout probability. Defaults to 0.0.
        proj_drop (float): Proj dropout layer. Defaults to 0.0.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 mixer: str = 'Global',
                 input_shape: Tuple[int, int] = (8, 25),
                 local_k: Tuple[int, int] = (7, 11),
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super().__init__()
        
        assert mixer in {'Global', 'Local'}, \
            "The type of mixer must belong to {'Global', 'Local'}"
        
        self.num_heads = num_heads
        self.mixer = mixer
        self.input_shape = input_shape
        
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim**-0.5
        
        # QKV projection
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        
        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

        if input_shape is not None:
            height, width = input_shape
            self.input_size = height * width
            self.embed_dims = embed_dims
            
        # Local attention mask
        if mixer == 'Local' and input_shape is not None:
            height, width = input_shape
            hk, wk = local_k
            
            # Create attention mask for local attention
            mask = torch.ones(
                [height * width, height + hk - 1, width + wk - 1],
                dtype=torch.float32)
            
            # Set local window to 0 (allowing attention)
            for h in range(height):
                for w in range(width):
                    mask[h * width + w, h:h + hk, w:w + wk] = 0.
            
            # Crop the mask to the correct size and set non-local attention to -inf
            mask = mask[:, hk // 2:height + hk // 2,
                       wk // 2:width + wk // 2].flatten(1)
            mask[mask >= 1] = -float('inf')
            
            # Register the mask as a buffer (persistent state)
            self.register_buffer('mask', mask[None, None, :, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, H*W, C)`.

        Returns:
            torch.Tensor: A tensor of shape :math:`(N, H*W, C)`.
        """
        if self.input_shape is not None:
            input_size, embed_dims = self.input_size, self.embed_dims
        else:
            _, input_size, embed_dims = x.shape

        # Project input to q, k, v
        qkv = self.qkv(x).reshape(-1, input_size, 3, self.num_heads,
                                 embed_dims // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        # Compute attention scores
        attn = q @ k.transpose(-2, -1)
        
        # Apply local attention mask if using Local mixer
        if self.mixer == 'Local':
            attn = attn + self.mask

        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(-1, input_size, embed_dims)
        
        # Final projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class MLP(nn.Module):
    """The MLP block.

    Args:
        in_features (int): The input features.
        hidden_features (int, optional): The hidden features.
            Defaults to None (same as in_features).
        out_features (int, optional): The output features.
            Defaults to None (same as in_features).
        drop (float, optional): Dropout probability. Defaults to 0.0.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 drop: float = 0.):
        super().__init__()
        
        # Set hidden and output features if not provided
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        # Define layers
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): A Tensor of shape :math:`(N, *, C)` where
                * means any number of additional dimensions and C is in_features.

        Returns:
            torch.Tensor: A tensor of shape :math:`(N, *, C)` where
                C is out_features.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MixingBlock(nn.Module):
   """The Mixing block.

   Args:
       embed_dims (int): Number of character components.
       num_heads (int): Number of heads
       mixer (str, optional): The mixer type. Defaults to 'Global'.
       window_size (Tuple[int ,int], optional): Local window size.
           Defaults to (7, 11).
       input_shape (Tuple[int, int], optional): The shape of input [H, W].
           Defaults to (8, 25).
       mlp_ratio (float, optional): The ratio of hidden features to input.
           Defaults to 4.0.
       qkv_bias (bool, optional): Whether a additive bias is required.
           Defaults to False.
       qk_scale (float, optional): A scaling factor. Defaults to None.
       drop (float, optional): Dropout rate. Defaults to 0.
       attn_drop (float, optional): Attention dropout rate. Defaults to 0.0.
       drop_path (float, optional): The probability of drop path.
           Defaults to 0.0.
       prenorm (bool, optional): Whether to normalize before mixing.
           Defaults to True.
   """

   def __init__(self,
                embed_dims: int,
                num_heads: int,
                mixer: str = 'Global',
                window_size: Tuple[int, int] = (7, 11),
                input_shape: Tuple[int, int] = (8, 25),
                mlp_ratio: float = 4.,
                qkv_bias: bool = False,
                qk_scale: float = None,
                drop: float = 0.,
                attn_drop: float = 0.,
                drop_path: float = 0.,
                prenorm: bool = True):
       super().__init__()

       # Layer normalization
       self.norm1 = nn.LayerNorm(embed_dims, eps=1e-6)
       self.norm2 = nn.LayerNorm(embed_dims, eps=1e-6)
       
       # Select mixer type
       if mixer in {'Global', 'Local'}:
           self.mixer = AttnMixer(
               embed_dims,
               num_heads=num_heads,
               mixer=mixer,
               input_shape=input_shape,
               local_k=window_size,
               qkv_bias=qkv_bias,
               qk_scale=qk_scale,
               attn_drop=attn_drop,
               proj_drop=drop)
       elif mixer == 'Conv':
           self.mixer = ConvMixer(
               embed_dims,
               num_heads=num_heads,
               input_shape=input_shape,
               local_k=window_size)
       else:
           raise TypeError('The mixer must be one of [Global, Local, Conv]')

       # Drop path
       self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
       
       # MLP block
       mlp_hidden_dim = int(embed_dims * mlp_ratio)
       self.mlp = MLP(
           in_features=embed_dims,
           hidden_features=mlp_hidden_dim,
           drop=drop)
       
       self.prenorm = prenorm

   def forward(self, x: torch.Tensor) -> torch.Tensor:
       """Forward function.

       Args:
           x (torch.Tensor): A Tensor of shape :math:`(N, H*W, C)`.

       Returns:
           torch.Tensor: A tensor of shape :math:`(N, H*W, C)`.
       """
       if self.prenorm:
           # Pre-normalization: norm -> mixing -> residual
           x = self.norm1(x + self.drop_path(self.mixer(x)))
           x = self.norm2(x + self.drop_path(self.mlp(x)))
       else:
           # Post-normalization: mixing -> norm -> residual
           x = x + self.drop_path(self.mixer(self.norm1(x)))
           x = x + self.drop_path(self.mlp(self.norm2(x)))
       return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path(x, self.drop_prob, self.training)
    
    def drop_path(self, x: torch.Tensor, drop_prob: float = 0., training: bool=False) -> torch.Tensor:
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        # handle tensors with different dimensions, not just 4D tensors.
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device)
        output = x.div(keep_prob) * random_tensor.floor()
        return output

class MergingBlock(nn.Module):
   """The last block of any stage, except for the last stage.

   Args:
       in_channels (int): The channels of input.
       out_channels (int): The channels of output.
       types (str, optional): Which downsample operation of ['Pool', 'Conv'].
           Defaults to 'Pool'.
       stride (Union[int, Tuple[int, int]], optional): Stride of the Conv.
           Defaults to (2, 1).
       act (Type[nn.Module], optional): Activation function class. Defaults to None.
   """

   def __init__(self,
                in_channels: int,
                out_channels: int,
                types: str = 'Pool',
                stride: Union[int, Tuple[int, int]] = (2, 1),
                act: Optional[Type[nn.Module]] = None):
       super().__init__()
       
       self.types = types
       if isinstance(stride, int):
           stride = (stride, stride)
           
       if types == 'Pool':
           self.avgpool = nn.AvgPool2d(
               kernel_size=(3, 5),
               stride=stride,
               padding=(1, 2))
           self.maxpool = nn.MaxPool2d(
               kernel_size=(3, 5),
               stride=stride,
               padding=(1, 2))
           self.proj = nn.Linear(in_channels, out_channels)
       else:
           self.conv = nn.Conv2d(
               in_channels,
               out_channels,
               kernel_size=3,
               stride=stride,
               padding=1)
           
       self.norm = nn.LayerNorm(out_channels)
       self.act = act() if act is not None else None

   def forward(self, x: torch.Tensor) -> torch.Tensor:
       """Forward function.

       Args:
           x (torch.Tensor): A Tensor of shape :math:`(N, C, H, W)`.

       Returns:
           torch.Tensor: A Tensor of shape :math:`(N, H/2*W, 2C)`.
       """
       if self.types == 'Pool':
           # Combine average and max pooling
           x = (self.avgpool(x) + self.maxpool(x)) * 0.5
           # Reshape: (N, C, H, W) -> (N, C, H*W) -> (N, H*W, C)
           out = self.proj(x.flatten(2).permute(0, 2, 1))
       else:
           # Apply convolution
           x = self.conv(x)
           # Reshape: (N, C, H, W) -> (N, C, H*W) -> (N, H*W, C)
           out = x.flatten(2).permute(0, 2, 1)
           
       # Apply normalization and optional activation
       out = self.norm(out)
       if self.act is not None:
           out = self.act(out)

       return out

class SVTREncoder(nn.Module):
    """A PyTorch implementation of SVTR (Scene Text Recognition with a Single Visual Model).
    
    Paper reference: https://arxiv.org/abs/2205.00159
    """

    def __init__(self,
                 img_size: Tuple[int, int] = (32, 100),
                 in_channels: int = 3,
                 embed_dims: Tuple[int, int, int] = (64, 128, 256),
                 depth: Tuple[int, int, int] = (3, 6, 3),
                 num_heads: Tuple[int, int, int] = (2, 4, 8),
                 mixer_types: Tuple[str] = ('Local',) * 6 + ('Global',) * 6,
                 window_size: Tuple[Tuple[int, int]] = ((7, 11), (7, 11), (7, 11)),
                 merging_types: str = 'Conv',
                 mlp_ratio: int = 4,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 drop_rate: float = 0.,
                 last_drop: float = 0.1,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 out_channels: int = 192,
                 max_seq_len: int = 25,
                 num_layers: int = 2,
                 prenorm: bool = True):
        super().__init__()
        
        self.img_size = img_size
        self.embed_dims = embed_dims
        self.out_channels = out_channels
        self.prenorm = prenorm
        
        # Patch Embedding
        self.patch_embed = OverlapPatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims[0],
            num_layers=num_layers)

        # Calculate number of patches and input shape
        num_patches = (img_size[1] // (2**num_layers)) * (img_size[0] // (2**num_layers))
        self.input_shape = [
            img_size[0] // (2**num_layers), 
            img_size[1] // (2**num_layers)
        ]

        # Position embedding
        # self.absolute_pos_embed = nn.Parameter(
        #     torch.zeros(1, num_patches, embed_dims[0]))
        # self.pos_drop = nn.Dropout(drop_rate)
        self.absolute_pos_embed = nn.Parameter(
            torch.zeros([1, num_patches, embed_dims[0]], dtype=torch.float32),
            requires_grad=True)
        self.pos_drop = nn.Dropout(drop_rate)


        # Calculate drop path rate for each block
        dpr = np.linspace(0, drop_path_rate, sum(depth))

        # Stage 1
        self.blocks1 = nn.ModuleList([
            MixingBlock(
                embed_dims=embed_dims[0],
                num_heads=num_heads[0],
                mixer=mixer_types[0:depth[0]][i],
                window_size=window_size[0],
                input_shape=self.input_shape,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[0:depth[0]][i],
                prenorm=prenorm) 
            for i in range(depth[0])
        ])

        # Downsample 1
        self.downsample1 = MergingBlock(
            in_channels=embed_dims[0],
            out_channels=embed_dims[1],
            types=merging_types,
            stride=(2, 1))
        
        input_shape = [self.input_shape[0] // 2, self.input_shape[1]]

        # Stage 2
        self.blocks2 = nn.ModuleList([
            MixingBlock(
                embed_dims=embed_dims[1],
                num_heads=num_heads[1],
                mixer=mixer_types[depth[0]:depth[0] + depth[1]][i],
                window_size=window_size[1],
                input_shape=input_shape,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0]:depth[0] + depth[1]][i],
                prenorm=prenorm)
            for i in range(depth[1])
        ])

        # Downsample 2
        self.downsample2 = MergingBlock(
            in_channels=embed_dims[1],
            out_channels=embed_dims[2],
            types=merging_types,
            stride=(2, 1))
        
        input_shape = [self.input_shape[0] // 4, self.input_shape[1]]

        # Stage 3
        self.blocks3 = nn.ModuleList([
            MixingBlock(
                embed_dims=embed_dims[2],
                num_heads=num_heads[2],
                mixer=mixer_types[depth[0] + depth[1]:][i],
                window_size=window_size[2],
                input_shape=input_shape,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0] + depth[1]:][i],
                prenorm=prenorm)
            for i in range(depth[2])
        ])

        # Final layers
        self.layer_norm = nn.LayerNorm(self.embed_dims[-1], eps=1e-6)
        self.avgpool = nn.AdaptiveAvgPool2d((1, max_seq_len))
        self.last_conv = nn.Conv2d(
            in_channels=embed_dims[2],
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False)
        self.hardswish = nn.Hardswish()
        self.dropout = nn.Dropout(p=last_drop)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        # Initialize position embedding
        nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)
        
        # Initialize other components
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function before the final pooling and projection."""
        # Patch embedding
        x = self.patch_embed(x)
        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # Stage 1
        for blk in self.blocks1:
            x = blk(x)
        x = self.downsample1(
            x.permute(0, 2, 1).reshape(-1, self.embed_dims[0], 
                                     self.input_shape[0], self.input_shape[1]))

        # Stage 2
        for blk in self.blocks2:
            x = blk(x)
        x = self.downsample2(
            x.permute(0, 2, 1).reshape(-1, self.embed_dims[1], 
                                     self.input_shape[0] // 2, self.input_shape[1]))

        # Stage 3
        for blk in self.blocks3:
            x = blk(x)
            
        if not self.prenorm:
            x = self.layer_norm(x)
            
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        x = self.forward_features(x)
        
        # Final processing
        x = self.avgpool(
            x.permute(0, 2, 1).reshape(-1, self.embed_dims[2], 
                                     self.input_shape[0] // 4, self.input_shape[1]))
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        
        return x