import torch
import torch.nn as nn

""" Class for double-convolution model used inside UNet architecture.
    Specifically, the implemented UNet network architecture utilizes padded 
    convolution, therefore the padding of the double-layer convolutions is set to 1. """
class DoubleConv(nn.Module):
    """ Initialize configurations. """
    def __init__(self, args, input_channels, output_channels):
        super(DoubleConv, self).__init__()
        # different possible configuration of double-conv block
        if args.use_batch_norm == True:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )
        elif args.use_double_batch_norm == True:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )
            """ In general, instance normalization does not require a 
                bias term because it already shifts the distribution of 
                the activations to have zero mean and unit variance. Therefore, 
                adding a bias term may not be necessary and could lead to overfitting. """
        elif args.use_inst_norm == True:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(output_channels, affine=True),
                nn.ReLU()
            )
        elif args.use_double_inst_norm == True:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(output_channels, affine=True),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(output_channels, affine=True),
                nn.ReLU()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels,
                          kernel_size=3, padding=1),
                nn.ReLU()
            )

    """ Method used to define the computations needed to create 
        the output of the double-convolution module given its input. """
    def forward(self, x):
        return self.conv(x)


""" Class for customizable UNet architecture: that allows for 
    choosing between different types of normalization (Batch or Instance), 
    as well as the U-depth and the number of filters (neurons) in each layer."""
class UNET(nn.Module):
    """ Initialize configurations. """
    def __init__(self, args, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        # contracting-path
        self.downs = nn.ModuleList()
        # expanding-path
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.args = args

        # down-part of UNET
        for feature in features:
            self.downs.append(DoubleConv(self.args, in_channels, feature))
            in_channels = feature

        # up-part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(self.args, feature*2, feature))

        # bottleneck of UNET
        self.bottleneck = DoubleConv(self.args, features[-1], features[-1]*2)
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1, padding=0)
        
        self.output_activation = nn.Sigmoid()

        if self.args.weights_init == True:
            self.initialize_weights()

    """ Method used to initialize the weights of the network. """
    def initialize_weights(self):
        print('\nPerforming weights initialization...')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    """ Method used to define the computations needed to 
        create the output of the neural network given its input. """
    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # reverses the order of elements in the skip_connections
        # starting from the bottom, the first skip connection to be merged will be the last one that was added.
        skip_connections = skip_connections[::-1]

        # step of 2 because ups contains both ConvTranspose2d and Conv2d modules 
        # (which are at indices 1, 3, ...)
        for idx in range(0, len(self.ups), 2):
            # ConvTranspose2d
            x = self.ups[idx](x)
            # len(skip_connections) = len(self.ups)/2
            skip_connection = skip_connections[idx//2]

            concat_skip = torch.cat((skip_connection, x), dim=1)

            # Conv2d
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)

        # remember: 
        # arc_change_net can work both 
        # with bcewl_loss and dc_loss/jac_loss
        if self.args.loss != 'bcewl_loss':
            x = self.output_activation(x)

        return x
    

class AutoEncoder(nn.Module):
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
            nn.ConvTranspose2d(in_channels=8 * hidden_dim, out_channels=1, kernel_size=2, stride=2)
        )

    def initialize_weights(self):
        print('\nPerforming weights initialization...')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        # print(f"\nencoder-shape: {x.shape}")

        x = self.decoder(x)
        # print(f"\ndecoder-shape: {x.shape}")

        return x
    

class ConvAutoencoder(torch.nn.Module):
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
