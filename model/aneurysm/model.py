import torch.nn as nn
import torch


class AneurysmSimple3DCNN(nn.Module):
    """Small, efficient 3D CNN for scan-level classification from patches.

    Architecture: Conv3d -> ReLU -> MaxPool -> Conv3d -> ReLU -> MaxPool -> Conv3d -> ReLU -> GlobalAvgPool -> FC
    Returns a single logit per example.
    """

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class AneurysmTinyUNet3D(nn.Module):
    """Tiny 3D U-Net style encoder-decoder repurposed for classification.

    Designed to be computationally lightweight: 2-level encoder, bottleneck, 2-level decoder,
    followed by global pooling and a classification head producing one logit.
    """

    def __init__(self, in_channels=1, out_channels=1, base_filters=8):
        super().__init__()

        # -------------------------
        # ENCODER (Downsampling)
        # -------------------------
        # First conv block
        # Input channels -> 8 feature maps
        self.enc1 = self.conv_block(in_channels, base_filters)

        # Second conv block (after maxpool)
        # 8 -> 16 feature maps
        self.enc2 = self.conv_block(base_filters, base_filters * 2)

        # Max pooling layer to downsample the volume
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # -------------------------
        # BOTTLENECK
        # -------------------------
        # This is the "deepest" part of the network
        # 16 -> 32 feature maps
        self.bottleneck = self.conv_block(base_filters * 2, base_filters * 4)

        # -------------------------
        # DECODER (Upsampling)
        # -------------------------
        # Upsample: 32 -> 16 feature maps
        # ConvTranspose3d increases the spatial size
        self.up2 = nn.ConvTranspose3d(
            base_filters * 4, base_filters * 2, kernel_size=2, stride=2
        )
        # Decoder conv block after concatenation
        self.dec2 = self.conv_block(base_filters * 4, base_filters * 2)

        # Upsample: 16 -> 8 feature maps
        self.up1 = nn.ConvTranspose3d(
            base_filters * 2, base_filters, kernel_size=2, stride=2
        )
        self.dec1 = self.conv_block(base_filters * 2, base_filters)

        # Instead of segmentation output → classification head
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # (B,8,1,1,1)
        self.fc = nn.Linear(base_filters, 1)        # single probability

    # ---------------------------------------------------------
    # Helper: Two simple Conv3D layers with ReLU
    # ---------------------------------------------------------
    def conv_block(self, in_ch, out_ch):
        """Two Conv3D layers with ReLU activations."""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    # ---------------------------------------------------------
    # FORWARD PASS
    # ---------------------------------------------------------
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # Shape: (B, base_filters, D, H, W)
        e2 = self.enc2(self.pool(e1))  # Shape: (B, base_filters*2, D/2, H/2, W/2)

        # Bottleneck
        b = self.bottleneck(self.pool(e2))  # Shape: (B, base_filters*4, ...)

        # Decoder level 2
        d2 = self.up2(b)  # Upsample: (B,16, D/2, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)  # Skip connection
        d2 = self.dec2(d2)

        # Decoder level 1
        d1 = self.up1(d2)  # Upsample: (B,8, D, H, W)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Classification head
        pooled = self.global_pool(d1).view(x.size(0), -1) # (B,8)
        logit = self.fc(pooled)
        return logit
