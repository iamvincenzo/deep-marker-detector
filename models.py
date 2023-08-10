import torch
import torch.nn as nn

""" Class for CAE architecture with ConvTranspose2d layers. """   
class AutoEncoder(nn.Module):
    """ Initialize Configurations. """
    def __init__(self, in_channels, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8 * hidden_dim, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(8 * hidden_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=8 * hidden_dim, out_channels=2 * 8 * hidden_dim, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(2 * 8 * hidden_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=2 * 8 * hidden_dim, out_channels=4 * 8 * hidden_dim, kernel_size=4, padding=1, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4 * 8 * hidden_dim, out_channels=2 * 8 * hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(2 * 8 * hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=2 * 8 * hidden_dim, out_channels=8 * hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(8 * hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8 * hidden_dim, out_channels=1, kernel_size=2, stride=2),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(f"\nencoder-shape: {x.shape}")

        x = self.decoder(x)
        # print(f"\ndecoder-shape: {x.shape}")

        return x

""" Class for CAE architecture with Upsample layers. """   
class ConvAutoencoder(torch.nn.Module):
    """ Initialize configurations. """
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=1),
            torch.nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=1)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=1, mode="nearest"),
            torch.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=1, mode="nearest"),
            torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=2),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
