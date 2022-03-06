
from paddle.nn import Layer
from paddle.nn import Conv2D, BatchNorm2D, ReLU6, Sequential

class Conv_bn(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 act=True):
        super(Conv_bn, self).__init__()
        self.conv = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, dilation=dilation, padding=padding, groups=groups)

        self.bn = BatchNorm2D(num_features=out_channels)
        if act:
            self.relu = ReLU6()
        else:
            self.relu = None

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Block(Layer):
    def __init__(self, in_channels, out_channels, dilation=1, stride=1):
        super(Block, self).__init__()

        if stride!=1 or in_channels!=out_channels:
            self.shortcut = Conv_bn(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=1, stride=stride, act=False)
        else:
            self.shortcut = None
        self.conv = Sequential(
            # 第一个
            Conv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=dilation, dilation=dilation,
                    groups=in_channels), #depthwise
            Conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1), #pointwise
            # 第二个
            Conv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=dilation, dilation=dilation,
                    groups=out_channels), #depthwise
            Conv_bn(in_channels=out_channels, out_channels=out_channels, kernel_size=1),  # pointwise
            # 第三个
            Conv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=dilation,
                   dilation=dilation, groups=out_channels), #depthwise
            Conv_bn(in_channels=out_channels, out_channels=out_channels, kernel_size=1),  # pointwise

        )

    def forward(self, inputs):
        x = self.conv(inputs)
        if self.shortcut is not None:
            x += self.shortcut(inputs)
        else:
            x += inputs
        return x

class Xception(Layer):
    def __init__(self):
        super(Xception, self).__init__()
        # Entry Flow
        self.layer1 = Sequential(
            Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        )
        self.layer2 = Block(64, 128, stride=2)
        self.layer3 = Block(128, 256, stride=2)
        self.layer4 = Block(256, 728, dilation=1, stride=2)

        midflow = []
        for i in range(17):
            midflow.append(Block(728, 728, dilation=2, stride=1))
        self.layer5 = Sequential(*midflow)

        self.exitflow = Sequential(
            Block(728, 1024, dilation=2, stride=1),
            Conv2D(in_channels=1024, out_channels=1024, kernel_size=3, padding=2, dilation=2, groups=1024), #depthwise
            Conv_bn(in_channels=1024, out_channels=1536, kernel_size=1), #pointwise

            Conv2D(in_channels=1536, out_channels=1536, kernel_size=3, padding=2, dilation=2, groups=1536), #depthwise
            Conv_bn(in_channels=1536, out_channels=1536, kernel_size=1),  # pointwise

            Conv2D(in_channels=1536, out_channels=1536, kernel_size=3, padding=2, dilation=2, groups=1536), #depthwise
            Conv2D(in_channels=1536, out_channels=2048, kernel_size=1) # pointwise
        )

    def forward(self, inputs):
        # Entry Flow
        x = self.layer1(inputs) #1/2
        lowlevel1 = x
        print(x.shape)
        x = self.layer2(x) #1/4
        lowlevel2 = x
        print(x.shape)
        x = self.layer3(x) #1/8
        print(x.shape)
        x = self.layer4(x) # 1/16
        print(x.shape)
        # Middle Flow
        x = self.layer5(x) # 1/16
        print(x.shape)
        # Exit Flow
        x = self.exitflow(x) #1/16
        print(x.shape)

        return x, lowlevel2


def main():
    import paddle
    data = paddle.rand((1,3,512,512))
    model = Xception()
    pred = model(data)

if __name__ == '__main__':
    main()