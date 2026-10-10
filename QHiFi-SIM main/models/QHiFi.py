import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU(alpha=0.45)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.activation(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.in_channels = in_channels
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction_ratio),
            nn.GELU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        x_fft = fft.fft2(x, dim=(-2, -1))
        x_fft = fft.fftshift(x_fft)
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        magnitude_pool = torch.mean(magnitude, dim=(-2, -1))
        phase_pool = torch.mean(phase, dim=(-2, -1))
        combined = torch.cat([magnitude_pool, phase_pool], dim=1)
        attention = self.fc(combined).view(b, c, 1, 1)
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(combined)
        return x * self.sigmoid(attention)


class FixedConv2d(nn.Module):
    def __init__(self, stride=1):
        super(FixedConv2d, self).__init__()
        self.stride = stride

    def forward(self, x, kernel_tensor):
        with torch.no_grad():
            k = kernel_tensor.detach()[0:1]
        kh, kw = k.shape[2], k.shape[3]

        in_ch = x.size(1)
        k = k.expand(in_ch, 1, kh, kw).contiguous()
        return F.conv2d(x, k, stride=self.stride, padding=(kh // 2, kw // 2), groups=in_ch)


class QHiFi(nn.Module):
    # ===== 修改区域 START：__init__ 新增 5 个消融控制参数 =====
    def __init__(self, in_channels=1, base_channels=16, num_blocks=6,
                 disable_ssfe=False, disable_wife=False,
                 disable_channel_attn=False, disable_spatial_attn=False,
                 disable_identity=False):
        # ===== 修改区域 END =====
        super().__init__()

        # ===== 修改区域 START：将开关保存为成员变量 =====
        self.disable_ssfe = disable_ssfe
        self.disable_wife = disable_wife
        self.disable_channel_attn = disable_channel_attn
        self.disable_spatial_attn = disable_spatial_attn
        self.disable_identity = disable_identity
        # ===== 修改区域 END =====

        self.fixed_conv = FixedConv2d(stride=1)
        self.sim_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ELU(alpha=1.0)
        )

        self.wf_aaa_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels * 2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),

            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU()
        )

        # ===== 修改区域 START：根据 disable_channel_attn 动态构建残差块 =====
        self.residual_blocks = nn.ModuleList()
        for i in range(num_blocks):
            if i % 2 == 0 and not self.disable_channel_attn:
                self.residual_blocks.append(
                    nn.Sequential(
                        ResidualBlock(base_channels * 2, base_channels * 2),
                        ChannelAttention(base_channels * 2)
                    )
                )
            else:
                self.residual_blocks.append(
                    ResidualBlock(base_channels * 2, base_channels * 2)
                )
        # ===== 修改区域 END =====

        self.spatial_attention = SpatialAttention()
        self.reconstruct = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
            nn.GELU()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out',
                    nonlinearity='leaky_relu', a=0.45)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out',
                    nonlinearity='leaky_relu', a=0.45
                )
                nn.init.constant_(m.bias, 0)

    def forward(self, x_sim, x_wf, psf):
        identity = x_sim
        x_1 = self.fixed_conv(x_sim, psf)

        # ===== 修改区域 START：SSFE 分支（可禁用） =====
        if self.disable_ssfe:
            # 用零张量占位，保持拼接时的通道数一致
            sim_features = torch.zeros(
                x_sim.shape[0], 16, x_sim.shape[2], x_sim.shape[3],
                device=x_sim.device, dtype=x_sim.dtype
            )
        else:
            sim_features = self.sim_feature_extractor(x_sim)
        # ===== 修改区域 END =====

        # ===== 修改区域 START：WIFE 分支（可禁用） =====
        if self.disable_wife:
            wf_1_features = torch.zeros(
                x_sim.shape[0], 16, x_sim.shape[2], x_sim.shape[3],
                device=x_sim.device, dtype=x_sim.dtype
            )
        else:
            wf_1 = torch.cat([x_wf, x_1], dim=1)
            wf_1_features = self.wf_aaa_feature_extractor(wf_1)
        # ===== 修改区域 END =====

        combined_features = torch.cat([sim_features, wf_1_features], dim=1)

        for block in self.residual_blocks:
            combined_features = block(combined_features)

        # ===== 修改区域 START：空间注意力（可禁用） =====
        if self.disable_spatial_attn:
            attended_features = combined_features
        else:
            attended_features = self.spatial_attention(combined_features)
        # ===== 修改区域 END =====

        output = self.reconstruct(attended_features)

        # ===== 修改区域 START：全局残差连接（可禁用） =====
        if not self.disable_identity:
            output = output + identity
        # ===== 修改区域 END =====

        min_val = output.view(output.size(0), 1, -1).min(dim=2, keepdim=True)[0].unsqueeze(3)
        max_val = output.view(output.size(0), 1, -1).max(dim=2, keepdim=True)[0].unsqueeze(3)
        output = (output - min_val) / (max_val - min_val + 1e-8)
        degraded_corrected = self.fixed_conv(output, psf)
        return output, degraded_corrected, x_1

