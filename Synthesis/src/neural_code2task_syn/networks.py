import torch
import torch.nn.functional as F
from torch import nn


class SimpleFullyConnectedNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleFullyConnectedNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class HandmadeFeaturesNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(HandmadeFeaturesNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.net = SimpleFullyConnectedNetwork(input_size, output_size)

    def forward(self, x):
        x = x['features']

        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, kernel_size, in_feats):
        """
        kernel_size: width of the kernels
        in_feats: number of channels in inputs
        """
        super(ResBlock, self).__init__()
        self.feat_size = in_feats
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.conv2 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.conv3 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out += residual
        out = self.relu(out)

        return out


class GridEncoder(nn.Module):
    def __init__(self,
                 kernel_size,
                 conv_stack,
                 fc_stack,
                 img_size):
        """
        kernel_size: width of the kernels
        conv_stack: Number of channels at each point of the convolutional part of
                    the network (includes the input)
        fc_stack: number of channels in the fully connected part of the network
        """
        super(GridEncoder, self).__init__()
        self.conv_layers = []
        for i in range(1, len(conv_stack)):
            if conv_stack[i - 1] != conv_stack[i]:
                block = nn.Sequential(
                    ResBlock(kernel_size, conv_stack[i - 1]),
                    nn.Conv2d(conv_stack[i - 1], conv_stack[i],
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2),
                    nn.ReLU(inplace=True)
                )
            else:
                block = ResBlock(kernel_size, conv_stack[i - 1])
            self.conv_layers.append(block)
            self.add_module("ConvBlock-" + str(i - 1), self.conv_layers[-1])

        # We have operated so far to preserve all of the spatial dimensions so
        # we can estimate the flattened dimension.
        first_fc_dim = conv_stack[-1] * img_size[-1] * img_size[-2]
        adjusted_fc_stack = [first_fc_dim] + fc_stack
        self.fc_layers = []
        for i in range(1, len(adjusted_fc_stack)):
            self.fc_layers.append(nn.Linear(adjusted_fc_stack[i - 1],
                                            adjusted_fc_stack[i]))
            self.add_module("FC-" + str(i - 1), self.fc_layers[-1])

    def forward(self, x):
        """
        x: batch_size x channels x Height x Width
        """
        # batch_size = x.size(0)

        # Convolutional part
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Flatten for the fully connected part
        x = x.view(-1)
        # Fully connected part
        for i in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[i](x))
        x = self.fc_layers[-1](x)

        return x


class HandmadeFeaturesAndSymworldNetwork(nn.Module):
    def __init__(self,
                 kernel_size,
                 conv_stack,
                 fc_stack,
                 img_size,
                 features_size,
                 output_size):
        super(HandmadeFeaturesAndSymworldNetwork, self).__init__()
        self.output_size = output_size

        self.encoder = GridEncoder(kernel_size,
                                   conv_stack,
                                   fc_stack,
                                   img_size)

        self.net = SimpleFullyConnectedNetwork(fc_stack[-1] + features_size,
                                               output_size)

    def forward(self, x):
        features = x['features']
        symworld = x['symworld']

        symworld = self.encoder(symworld)
        x = torch.cat([features, symworld], dim=0)

        return self.net(x)
