# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Index",
    "ECA",
    "MCBAMChannelAttention",
    "MCBAM",
    "GSConv",
    "GnConv",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """
    Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """
        Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """
        Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """
        Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """
        Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """
    Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """
        Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """
        Apply activation and convolution transpose operation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """
    Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, W, H) and output shape is (B, 4C, W/2, H/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """
    Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = max(1, c2 // 2)  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """
        Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """
        Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """
        Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """
        Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """
        Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """
        Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """
    Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """
    Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """
        Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """
        Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """
    Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """
        Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """
        Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]

class ECA(nn.Module):
    """Efficient Channel Attention (ECA) Block.

    This module implements the ECA mechanism from the paper:
    "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"
    (Wang et al., CVPR 2020).

    The block adaptively determines the size of the 1D convolution kernel
    based on the number of channels in the input feature map, which models
    local cross-channel interactions efficiently without dimensionality
    reduction.

    Args:
        gamma (int, optional): Hyperparameter for controlling kernel size scaling.
            Default is 2.
        b (int, optional): Bias term in the kernel size formula.
            Default is 1.

    Shape:
        - Input: Tensor of shape (N, C, H, W)
        - Output: Tensor of shape (N, C, H, W), same as input

    Example:
        >>> eca = ECA(gamma=2, b=1)
        >>> x = torch.randn(8, 64, 32, 32)  # [N, C, H, W]
        >>> out = eca(x)
        >>> out.shape
        torch.Size([8, 64, 32, 32])
    """

    def __init__(self, gamma=2, b=1):
        super(ECA, self).__init__()
        self.gamma = gamma
        self.b = b
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        # Conv1d is built dynamically in forward() since kernel size depends on C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for ECA block.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W) with
            channel-wise attention applied.
        """
        N, C, H, W = x.size()

        # Dynamic kernel size calculation
        t = int(abs((math.log(C, 2) + self.b) / self.gamma))
        k = t if t % 2 else t + 1

        # Build 1D convolution dynamically
        conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False).to(x.device)

        # Global average pooling → [N, C, 1, 1]
        y = self.avg_pool(x)

        # Apply 1D conv along channel dimension
        y = conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)  # back to [N, C, 1, 1]

        # Attention weights
        y = self.sigmoid(y)

        # Reweight input
        return x * y.expand_as(x)

class MCBAMChannelAttention(nn.Module):
    """Multi-branch Channel Attention module (M-CBAM).

    Extends CBAM channel attention by adding a third branch formed from the
    sum of the average-pooled and max-pooled descriptors before applying
    the shared MLP. The outputs of all three branches are aggregated to
    generate richer channel attention weights.

    Args:
        channels (int): Number of input channels.
        reduction (int, optional): Reduction ratio for the MLP hidden layer.
            Defaults to 16.

    Attributes:
        mlp (nn.Sequential): Shared MLP with two 1×1 Conv layers and ReLU.
        avg_pool (nn.AdaptiveAvgPool2d): Global average pooling.
        max_pool (nn.AdaptiveMaxPool2d): Global max pooling.
        sigmoid (nn.Sigmoid): Activation function for attention weights.
    
    References: 
        - "A Lightweight Rice Pest Detection Algorithm Using Improved Attention Mechanism and YOLOv8" 
            (Yin et al., MDPI 2024) 
            https://www.mdpi.com/2077-0472/14/7/1052
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = channels // reduction

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Applies multi-branch channel attention.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with channel attention applied.
        """
        avg_vec = self.avg_pool(x)
        max_vec = self.max_pool(x)
        sum_vec = avg_vec + max_vec

        avg_out = self.mlp(avg_vec)
        max_out = self.mlp(max_vec)
        sum_out = self.mlp(sum_vec)

        att = avg_out + max_out + sum_out
        return x * self.act(att)

class MCBAM(nn.Module):
    """Multi-branch Convolutional Block Attention Module (M-CBAM).

    Extends CBAM by using a multi-branch channel attention mechanism
    combined with standard CBAM spatial attention.

    Args:
        channels (int): Number of input channels.
        reduction (int, optional): Reduction ratio for channel MLP.
            Defaults to 16.
        spatial_kernel (int, optional): Kernel size for spatial attention.
            Defaults to 7.

    Attributes:
        channel_att (MCBAMChannelAttention): Multi-branch channel attention module.
        spatial_att (SpatialAttention): Standard spatial attention module.
    
    References: 
        - "A Lightweight Rice Pest Detection Algorithm Using Improved Attention Mechanism and YOLOv8" 
            (Yin et al., MDPI 2024) 
            https://www.mdpi.com/2077-0472/14/7/1052
    """

    def __init__(self, c1, c2=None, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_att = MCBAMChannelAttention(
            c1, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        """Applies M-CBAM sequentially (channel → spatial).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after channel + spatial attention.
        """
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class GSConv(nn.Module):
    """
    GSConv: Grid Sensitive Convolution
    
    Architecture flow:
    1. Split input into two groups
    2. Conv on first group → C₂/2 channels
    3. Identity on second group → C₂/2 channels  
    4. Concat both groups → C₂ channels
    5. Channel shuffle for feature mixing
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k (int): Kernel size. Default: 1
        s (int): Stride. Default: 1
        g (int): Groups for convolution. Default: 1
        act (bool): Apply activation. Default: True
    
    References:
        - "Deep learning-based rice pest detection research"
            (Xiong et al., PLoS ONE 2024)
            https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0313387
        
        - "A lightweight YOLOv7 insulator defect detection algorithm based on DSC-SE"
            (Zhang et al., PLoS ONE 2023)
            https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0289162
    """
    
    def __init__(self, c1, c2, k=3, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2

        # If stride=2, we replace pooling with a stride-2 conv
        self.down = None
        if s == 2:
            self.down = Conv(c1, c1, k=3, s=2, p=1, act=act)

        # GSConv inner operations always use stride=1
        self.cv1 = Conv(c1 // 2, c_, k, 1, g=g, act=act)
        self.cv2 = Conv(c1 // 2, c_, 1, 1, act=act)

        self.s = s

    def forward(self, x):
        # Proper YOLO-compatible downsampling
        if self.s == 2:
            x = self.down(x)

        # GSConv pathway
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.cv1(x1)
        x2 = self.cv2(x2)

        out = torch.cat([x1, x2], dim=1)
        return self.channel_shuffle(out, 2)

    @staticmethod
    def channel_shuffle(x, groups=2):
        b, c, h, w = x.size()
        g = groups
        x = x.reshape(b, g, c // g, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.reshape(b, c, h, w)

class GnConv(nn.Module):
    """
    Recursive Gated Convolution (GnConv)
    
    Performs high-order spatial interactions with gated convolutions and recursive designs.
    
    Architecture:
    1. Project input to 2x channels
    2. Split into gating (pwa) and feature (abc) parts
    3. Apply depthwise conv with multi-scale processing
    4. Progressive pointwise convolutions with gating
    5. Final projection to output channels
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (if None, c2 = c1)
        order (int): Hierarchy order for multi-scale processing. Default: 5
        kernel (int): Kernel size for depthwise conv. Default: 7
        s (float): Scaling factor for dwconv output. Default: 1.0
        act (bool): Add activation (kept for compatibility). Default: False
    
    References:
        - "HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions"
            (Rao et al., NeurIPS 2022)
            https://papers.nips.cc/paper_files/paper/2022/file/436d042b2dd81214d23ae43eb196b146-Paper-Conference.pdf
    """
    
    def __init__(self, c1, c2=None, order=5, kernel=7, s=1.0, act=False):
        super().__init__()
        
        # Output channels default to input channels
        c2 = c2 or c1
        
        self.order = order
        self.scale = s
        
        # Calculate dimension hierarchy (from small to large)
        # dims[0] is smallest, dims[-1] is largest
        self.dims = [c1 // 2 ** i for i in range(order)]
        self.dims.reverse()  # Now: [c1/2^(order-1), ..., c1/4, c1/2]
        
        # Input projection: c1 → 2*c1
        # Split into: dims[0] for gating + sum(dims) for features
        self.proj_in = nn.Conv2d(c1, self.dims[0] + sum(self.dims), 1, bias=False)
        
        # Depthwise convolution (integrated, no separate function)
        self.dwconv = nn.Conv2d(
            sum(self.dims), 
            sum(self.dims), 
            kernel_size=kernel, 
            padding=(kernel - 1) // 2,
            bias=True,
            groups=sum(self.dims)  # Depthwise: each channel conv separately
        )
        
        # Progressive pointwise convolutions
        # Each pw conv expands from dims[i] → dims[i+1]
        self.pws = nn.ModuleList([
            nn.Conv2d(self.dims[i], self.dims[i + 1], 1, bias=False) 
            for i in range(order - 1)
        ])
        
        # Output projection: dims[-1] → c2
        self.proj_out = nn.Conv2d(self.dims[-1], c2, 1, bias=False)
        
    def forward(self, x):
        """
        Forward pass through GnConv
        
        Args:
            x (torch.Tensor): Input tensor [B, C1, H, W]
            
        Returns:
            torch.Tensor: Output tensor [B, C2, H, W]
        """
        # Input projection
        fused_x = self.proj_in(x)
        
        # Split into gating path (pwa) and feature path (abc)
        pwa, abc = torch.split(
            fused_x, 
            (self.dims[0], sum(self.dims)), 
            dim=1
        )
        
        # Depthwise convolution with scaling
        dw_abc = self.dwconv(abc) * self.scale
        
        # Split dwconv output into hierarchy
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        
        # First gating: pwa * dw_list[0]
        x = pwa * dw_list[0]
        
        # Progressive pointwise convolutions with gating
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
        
        # Final output projection
        x = self.proj_out(x)
        
        return x    
