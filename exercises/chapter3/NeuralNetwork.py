import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import copy


class elasticityDataset(Dataset):
    def __init__(self, device, numberOfSamples):
        self.E = torch.load("datasetStrain/E.pt", map_location=device, weights_only=False)[:numberOfSamples]
        self.eps = torch.load("datasetStrain/eps.pt", map_location=device, weights_only=False)[:numberOfSamples]
        self.numberOfSamples = len(self.E)

    def __len__(self):
        return self.numberOfSamples

    def __getitem__(self, idx):
        return (self.E[idx], self.eps[idx])


class noActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def makeCNNBlocks(channels, numberOfConvolutionsPerBlock, kernelSize, activation, normalization, skipChannels=None,
                  lastActivation=True):
    padding = (kernelSize - 1) // 2
    if skipChannels == None:
        skipChannels = [0 for i in channels]

    convolutions = torch.nn.ModuleList()
    normalizations = torch.nn.ModuleList()
    activations = torch.nn.ModuleList()
    for i in range(len(channels) - 1):
        for j in range(numberOfConvolutionsPerBlock):
            if j == 0:
                inChannels = channels[i] + skipChannels[i]
            else:
                inChannels = channels[i + 1]

            convolutions.append(torch.nn.Conv2d(inChannels,
                                                channels[i + 1],
                                                kernelSize,
                                                stride=1,
                                                padding=padding))
            normalizations.append(normalization(inChannels))

            if i == len(channels) - 2 and j == numberOfConvolutionsPerBlock - 1 and lastActivation == False:
                activations.append(noActivation())
            else:
                activations.append(activation())

    return convolutions, normalizations, activations


class UNet(torch.nn.Module):
    def __init__(self, channels, channelsOut, numberOfConvolutionsPerBlock, kernelSize):
        super().__init__()
        self.kernelSize = kernelSize
        self.channels = channels  # channelsIn is defined implicitly
        self.channelsOut = channelsOut
        self.numberOfConvolutionsPerBlock = numberOfConvolutionsPerBlock

        self.numberOfBottleNeckLayers = self.numberOfConvolutionsPerBlock
        self.channelsDown = copy.deepcopy(self.channels)
        self.channelsUp = copy.deepcopy(self.channels)[::-1]
        self.channelsUp[-1] = self.channelsOut

        self.activation = lambda: torch.nn.LeakyReLU(inplace=True)
        self.normalization = lambda s: torch.nn.BatchNorm2d(s)

        # downsampling
        self.convolutionsDown, self.normalizationsDown, self.activationsDown = makeCNNBlocks(self.channelsDown,
                                                                                             self.numberOfConvolutionsPerBlock,
                                                                                             self.kernelSize,
                                                                                             self.activation,
                                                                                             self.normalization)
        self.downsample = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.convolutionsBottleneck, self.normalizationsBottleneck, self.activationsBottleneck = makeCNNBlocks(
            [self.channelsDown[-1], self.channelsUp[0]],
            self.numberOfBottleNeckLayers,
            self.kernelSize,
            self.activation,
            self.normalization)

        # upsampling
        skipChannels = self.channelsDown[1:][::-1]
        skipChannels[-1] += 1
        self.convolutionsUp, self.normalizationsUp, self.activationsUp = makeCNNBlocks(self.channelsUp,
                                                                                       self.numberOfConvolutionsPerBlock,
                                                                                       self.kernelSize,
                                                                                       self.activation,
                                                                                       self.normalization,
                                                                                       skipChannels=skipChannels,
                                                                                       lastActivation=False)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')  # could be changed to 'nearest'

    def forward(self, x):
        x0 = x
        x_ = []  # skip connections

        for i in range(len(self.channelsDown) - 1):
            for j in range(self.numberOfConvolutionsPerBlock):
                index = i * self.numberOfConvolutionsPerBlock + j
                x = self.activationsDown[index](self.convolutionsDown[index](self.normalizationsDown[index](x)))
            x_.append(x)
            x = self.downsample(x)

        for j in range(self.numberOfBottleNeckLayers):
            index = j
            x = self.activationsBottleneck[index](
                self.convolutionsBottleneck[index](self.normalizationsBottleneck[index](x)))

        for i in range(len(self.channelsUp) - 1):
            x = torch.cat((self.upsample(x), x_[-(i + 1)]), 1)  # concatenate for skip connections
            if i == len(self.channelsUp) - 2:
                x = torch.cat((x, x0), 1)
            for j in range(self.numberOfConvolutionsPerBlock):
                index = i * self.numberOfConvolutionsPerBlock + j
                x = self.activationsUp[index](self.convolutionsUp[index](self.normalizationsUp[index](x)))

        return x


class FeedforwardCNN(torch.nn.Module):
    def __init__(self, channels, channelsOut, kernelSize):
        super().__init__()
        self.kernelSize = kernelSize
        self.channels = channels + [channelsOut]  # channelsIn is defined implicitly
        #        self.channelsOut = channelsOut

        self.activation = lambda: torch.nn.LeakyReLU(inplace=True)
        self.normalization = lambda s: torch.nn.BatchNorm2d(s)

        self.convolutions, self.normalizations, self.activations = makeCNNBlocks(self.channels, 1, kernelSize,
                                                                                 self.activation, self.normalization,
                                                                                 lastActivation=False)

    def forward(self, x):
        for i in range(len(self.channels) - 1):
            x = self.activations[i](self.convolutions[i](self.normalizations[i](x)))
        return x


class UNetWithSubsequentFeedforwardCNN(torch.nn.Module):
    def __init__(self, channelsUNet, numberOfConvolutionsPerBlockUNet, channelsFeedforwardCNN, channelsOut, kernelSize):
        super().__init__()
        self.uNet = UNet(channelsUNet, channelsFeedforwardCNN[0], numberOfConvolutionsPerBlockUNet, kernelSize)
        self.feedforwardCNN = FeedforwardCNN(channelsFeedforwardCNN, channelsOut, kernelSize)

        self.activation = torch.nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.uNet(x)
        x = self.activation(x)
        x = self.feedforwardCNN(x)
        return x


def initWeights(m):
    """Initialize weights of neural network with xavier initialization."""
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        m.bias.data.fill_(0.0)


def costFunction(prediction, label):
    return torch.mean((prediction - label) ** 2)
