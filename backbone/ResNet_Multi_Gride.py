


import paddle
from paddle.nn import Layer
from paddle.nn import Conv2D, BatchNorm2D, ReLU, MaxPool2D, Sequential


class Conv_bn(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, act=True):
        super(Conv_bn, self).__init__()
        self.conv = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, dilation=dilation, padding=padding)
                
        self.bn = BatchNorm2D(num_features=out_channels)
        if act:
            self.relu = ReLU()
        else:
            self.relu = None
    
    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Bottleneck(Layer):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, padding=1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.conv2 = Conv_bn(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, 
                dilation=dilation, padding=padding)
        self.conv3 = Conv_bn(in_channels=out_channels, out_channels=out_channels*4, kernel_size=1, act=False)
        self.conv4 = Conv_bn(in_channels=in_channels, out_channels=out_channels*4, kernel_size=1, stride=stride, act=False)
        self.relu = ReLU()

    def forward(self,inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x += self.conv4(inputs)
        x = self.relu(x)
        return x

class ResNet50(Layer):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = Conv_bn(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.pool1 = MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_block(in_channels=64, out_channels=64, stride=1, times=3)
        self.layer2 = self.make_block(in_channels=256, out_channels=128, stride=2, times=4)
        self.layer3 = self.make_block(in_channels=512, out_channels=256, stride=1, dilation=2, padding=2,  times=4)
        self.layer4 = self.make_block(in_channels=1024, out_channels=512, stride=1, dilation=2, padding=2, times=4, multi_gride=True)
        self.layer5 = self.make_block(in_channels=2048, out_channels=512, stride=1, dilation=4, padding=4, times=4, multi_gride=True)
        self.layer6 = self.make_block(in_channels=2048, out_channels=512, stride=1, dilation=8, padding=8, times=4, multi_gride=True)
        self.layer7 = self.make_block(in_channels=2048, out_channels=512, stride=1, dilation=16, padding=16, times=4, multi_gride=True)


    def forward(self,inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x

    def make_block(self, in_channels, out_channels, stride, dilation=1, padding=1, times=1, multi_gride=False):
        layers = []
        if multi_gride:
            for i in range(times):
                if i==0:
                    layers.append(
                        Bottleneck(in_channels=in_channels, out_channels=out_channels, stride=stride, dilation=dilation, padding=dilation)
                    )
                else:
                    layers.append(
                        Bottleneck(in_channels=out_channels*4, out_channels=out_channels, dilation=dilation*pow(2,i),
                         padding=dilation*pow(2,i))
                    )
            return Sequential(*layers)
        for i in range(times):
            if i==0:
                layers.append(
                    Bottleneck(in_channels=in_channels, out_channels=out_channels, stride=stride, dilation=dilation, padding=padding)
                )
            else:
                layers.append(
                    Bottleneck(in_channels=out_channels*4, out_channels=out_channels, dilation=dilation, padding=padding)
                )
        return Sequential(*layers)

def main():
    data = paddle.rand((1, 3, 512, 512))
    model = ResNet50()
    model.train()
    out = model(data)
    print(out.shape)

if __name__ == '__main__':
    main()


