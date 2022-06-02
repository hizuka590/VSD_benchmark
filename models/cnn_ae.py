import torch
import torch.nn as nn


##########################
### MODEL
##########################
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :416, :416]



class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()

        self.encoder = nn.Sequential(  # 784
            nn.Conv2d(3, 32, stride=(1, 1), kernel_size=(3, 3), padding=1), # 32 416 416
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),# 64 208 208
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),#64 104 104
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=1),# 32 52 52
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, stride=(2, 2), kernel_size=(3, 3), padding=1),  # 32 26 26
            nn.Flatten(),
            nn.Linear(21632, 256)
        )
        self.decoder = nn.Sequential(
            torch.nn.Linear(256, 21632),
            Reshape(-1, 32, 26, 26),
            nn.ConvTranspose2d(32, 32, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=0),
            Trim(),  # 1x29x29 -> 1x28x28
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def test():
    x = torch.randn((3, 3, 416, 416))
    model = Model()
    preds = model(x)
    print(preds)
    print(preds.max(), preds.min())
    assert preds.shape == (3,1,416,416)

if __name__ == "__main__":
    test()