import torch
import torch.nn as nn


class FireSmokeModel(nn.Module):
    class _TwoConv(nn.Module):
        def __init__(self, in_channel, out_channel):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channel),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channel),
            )

        def forward(self, x):
            return self.model(x)

    class _Encoder(nn.Module):
        def __init__(self, in_channel, out_channel):
            super().__init__()
            self.conv = FireSmokeModel._TwoConv(in_channel, out_channel)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            y = self.conv(x)
            x = self.pool(y)
            return x, y

    class _Decoder(nn.Module):
        def __init__(self, in_channel, out_channel):
            super().__init__()
            self.transpose = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
            self.conv = FireSmokeModel._TwoConv(in_channel, out_channel)

        def forward(self, x, y):
            x = self.transpose(x)
            x = torch.cat([x, y], dim=1)
            x = self.conv(x)
            return x

    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.encoder_1 = self._Encoder(in_channel, 64)
        self.encoder_2 = self._Encoder(64, 128)
        self.encoder_3 = self._Encoder(128, 256)
        self.encoder_4 = self._Encoder(256, 512)
        self.conv_transpose = self._TwoConv(512, 1024)
        self.decoder_1 = self._Decoder(1024, 512)
        self.decoder_2 = self._Decoder(512, 256)
        self.decoder_3 = self._Decoder(256, 128)
        self.decoder_4 = self._Decoder(128, 64)
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x, y1 = self.encoder_1(x)
        x, y2 = self.encoder_2(x)
        x, y3 = self.encoder_3(x)
        x, y4 = self.encoder_4(x)
        x = self.conv_transpose(x)
        x = self.decoder_1(x, y4)
        x = self.decoder_2(x, y3)
        x = self.decoder_3(x, y2)
        x = self.decoder_4(x, y1)
        x = self.out(x)
        return x
