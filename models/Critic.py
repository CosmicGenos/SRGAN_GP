import torch
import torch.nn as nn

'''
This is dicriminator model, it is a critic model, because it is used in WGAN-GP, 
This is easiest way since middle convolutional block in difference sizes.
this input must be 3*96*96 , if you want to any type of input use,  nn.AdaptiveAvgPool2d((1, 1)) to before fattening happens.
'''

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            self._block(64, 64, stride=2),
            self._block(64, 128, stride=1),
            self._block(128, 128, stride=2),
            self._block(128, 256, stride=1),
            self._block(256, 256, stride=2),
            self._block(256, 512, stride=1),
            self._block(512, 512, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )

    def _block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

if __name__ == '__main__':
    input = torch.randn((1, 3, 96, 96))
    discriminator = Critic()
    output = discriminator(input)
    print(output.shape)
    print(output)
    # Expected output: (1, 1)