import torch
import torch.nn as nn


# class UNet(nn.Module):
#     class _TwoConv(nn.Module):
#         def __init__(self, in_channel, out_channel):
#             super().__init__()
#             self.model = nn.Sequential(
#                 nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
#                 nn.ReLU(inplace=True),
#                 nn.BatchNorm2d(out_channel),
#                 nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
#                 nn.ReLU(inplace=True),
#                 nn.BatchNorm2d(out_channel),
#             )
#
#         def forward(self, x):
#             return self.model(x)
#
#     class _Encoder(nn.Module):
#         def __init__(self, in_channel, out_channel):
#             super().__init__()
#             self.conv = UNet._TwoConv(in_channel, out_channel)
#             self.pool = nn.MaxPool2d(2)
#
#         def forward(self, x):
#             y = self.conv(x)
#             x = self.pool(y)
#             return x, y
#
#     class _Decoder(nn.Module):
#         def __init__(self, in_channel, out_channel):
#             super().__init__()
#             self.transpose = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
#             self.conv = UNet._TwoConv(in_channel, out_channel)
#
#         def forward(self, x, y):
#             x = self.transpose(x)
#             x = torch.cat([x, y], dim=1)
#             x = self.conv(x)
#             return x
#
#     def __init__(self, num_classes):
#         super().__init__()
#         self.encoder_1 = self._Encoder(3, 64)
#         self.encoder_2 = self._Encoder(64, 128)
#         self.encoder_3 = self._Encoder(128, 256)
#         self.encoder_4 = self._Encoder(256, 512)
#         self.conv_transpose = self._TwoConv(512, 1024)
#         self.decoder_1 = self._Decoder(1024, 512)
#         self.decoder_2 = self._Decoder(512, 256)
#         self.decoder_3 = self._Decoder(256, 128)
#         self.decoder_4 = self._Decoder(128, 64)
#         self.out = nn.Conv2d(64, num_classes, kernel_size=1)
#
#     def forward(self, x):
#         x, y1 = self.encoder_1(x)
#         x, y2 = self.encoder_2(x)
#         x, y3 = self.encoder_3(x)
#         x, y4 = self.encoder_4(x)
#         x = self.conv_transpose(x)
#         x = self.decoder_1(x, y4)
#         x = self.decoder_2(x, y3)
#         x = self.decoder_3(x, y2)
#         x = self.decoder_4(x, y1)
#         x = self.out(x)
#         return x


class ResNetUNet(nn.Module):
    class _Block(nn.Module):
        def __init__(self, in_channel, out_channel, stride=1):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.conv(x)

    class _ConvDepthWise(nn.Module):
        def __init__(self, in_channel):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.conv(x)

    class _UNetBlock(nn.Module):
        def __init__(self, in_channel, out_channel):
            super().__init__()
            self.input = nn.Conv2d(in_channel, out_channel, kernel_size=1)
            self.conv = nn.ModuleList([ResNetUNet._ConvDepthWise(out_channel) for _ in range(6)])
            self.output = nn.ConvTranspose2d(out_channel, out_channel, kernel_size=2, stride=2)

        def forward(self, x):
            x = self.input(x)
            for layer in self.conv:
                x = layer(x)
            x = self.output(x)
            return x

    class _ConvBlock(nn.Module):
        def __init__(self, in_channel, out_channel, stride=2):
            super().__init__()
            self.bottleneck = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel * 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channel * 4),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * 4),
            )

        def forward(self, x):
            return torch.relu(self.bottleneck(x) + self.skip(x))

    class _IdenBlock(nn.Module):
        def __init__(self, in_channel, out_channel):
            super().__init__()
            self.bottleneck = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel * 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channel * 4),
            )

        def forward(self, x):
            return torch.relu(self.bottleneck(x) + x)


    def __init__(self, num_classes): # 3x1024x1024
        super().__init__()
        self.block_1_2 = nn.Sequential(
            self._Block(3, 32),
            self._Block(32, 64),
        )
        self.block_3 = self._Block(64, 128, stride=2) # 128x512x512 - y1
        self.max_pool = nn.MaxPool2d(2) # 128x256x256
        self.conv_block_0 = self._ConvBlock(128, 32, stride=1) #128x256x256
        self.iden_block_1_2 = nn.Sequential(
            self._IdenBlock(32 * 4, 32),
            self._IdenBlock(32 * 4, 32), #128x256x256 - y2
        )
        self.conv_block_3 = self._ConvBlock(32 * 4, 128)#512x128x128
        self.iden_block_4_6 = nn.Sequential(
            self._IdenBlock(128 * 4, 128),
            self._IdenBlock(128 * 4, 128),
            self._IdenBlock(128 * 4, 128), #512x128x128 - y3
        )
        self.conv_block_7 = self._ConvBlock(128 * 4, 256)#1024x64x64
        self.iden_block_8_22 = nn.ModuleList([self._IdenBlock(256 * 4, 256) for _ in range(15)])# - y4
        self.conv_block_23 = self._ConvBlock(256 * 4, 512) # 2048x32x32
        self.iden_block_24_25 = nn.Sequential(
            self._IdenBlock(512 * 4, 512),
            self._IdenBlock(512 * 4, 512),
        )
        self.unet_block_1 = self._UNetBlock(512 * 4, 512)
        self.unet_block_2 = self._UNetBlock(1536, 256)
        self.unet_block_3 = self._UNetBlock(768, 384)
        self.unet_block_4 = self._UNetBlock(512, 128)
        self.unet_block_5 = self._UNetBlock(256, 64)
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.block_1_2(x)
        y1 = self.block_3(x)
        x = self.max_pool(y1)
        x = self.conv_block_0(x)
        y2 = self.iden_block_1_2(x)
        x = self.conv_block_3(y2)
        y3 = self.iden_block_4_6(x)
        y4 = self.conv_block_7(y3)
        for block in self.iden_block_8_22:
            y4 = block(y4)
        x = self.conv_block_23(y4)
        x = self.iden_block_24_25(x)
        x = self.unet_block_1(x)
        x = torch.cat([x, y4], dim=1)
        x = self.unet_block_2(x)
        x = torch.cat([x, y3], dim=1)
        x = self.unet_block_3(x)
        x = torch.cat([x, y2], dim=1)
        x = self.unet_block_4(x)
        x = torch.cat([x, y1], dim=1)
        x = self.unet_block_5(x)
        x = self.out(x)
        return x
