
import paddle
from paddle.nn import Conv2D, BatchNorm2D, MaxPool2D, Conv2DTranspose, ReLU, Pad2D, Softmax

class conv_bn_relu(paddle.nn.Layer):
    def __init__(self, in_channels,out_channels, kernel_size=1, stride=1, padding=0):
        super(conv_bn_relu, self).__init__()
        self.conv = Conv2D(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = BatchNorm2D(out_channels)
        self.relu = ReLU()
    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Encoder(paddle.nn.Layer):
    def __init__(self,in_channels,out_channels):
        super(Encoder,self).__init__()

        self.conv1 = conv_bn_relu(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=3, stride=1, padding=1)
        self.conv2 = conv_bn_relu(in_channels=out_channels, out_channels=out_channels,
                                  kernel_size=3, stride=1, padding=1)
        self.pool = MaxPool2D(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self,inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x_pooled = self.pool(x)
        return x, x_pooled

class Decoder(paddle.nn.Layer):
    def __init__(self,in_channels,out_channels):
        super(Decoder,self).__init__()
        self.up = Conv2DTranspose(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=2, stride=2)

        self.conv1 = conv_bn_relu(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=3, stride=1, padding=1)
        self.conv2 = conv_bn_relu(in_channels=out_channels, out_channels=out_channels,
                                  kernel_size=3, stride=1, padding=1)
        self.pool = MaxPool2D(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, inputs_prev, inputs):
        x = self.up(inputs)
        h_diff = (inputs_prev.shape[2]-x.shape[2])
        w_diff = (inputs_prev.shape[3]-x.shape[3])
        x = Pad2D(padding=[h_diff//2, h_diff-h_diff//2,
                               w_diff//2, w_diff-w_diff//2])(x)
        x = paddle.concat([inputs_prev, x], axis=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UNet(paddle.nn.Layer):
    def __init__(self,num_classes):
        super(UNet, self).__init__()

        self.down1 = Encoder(in_channels=3, out_channels=64)
        self.down2 = Encoder(in_channels=64, out_channels=128)
        self.down3 = Encoder(in_channels=128, out_channels=256)
        self.down4 = Encoder(in_channels=256, out_channels=512)

        self.mid_conv1 = conv_bn_relu(in_channels=512, out_channels=1024)

        self.mid_conv2 = conv_bn_relu(in_channels=1024, out_channels=1024)

        self.up1 = Decoder(in_channels=1024, out_channels=512)
        self.up2 = Decoder(in_channels=512, out_channels=256)
        self.up3 = Decoder(in_channels=256, out_channels=128)
        self.up4 = Decoder(in_channels=128, out_channels=64)

        self.last_conv = Conv2D(in_channels=64, out_channels=num_classes,
                                kernel_size=1)

    def forward(self,inputs):
        #下采样层
        x1, x = self.down1(inputs) #其实x1=x，保存x后面需要
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)

        # mid层
        x = self.mid_conv1(x)
        x = self.mid_conv2(x)

        # 上采样层
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.last_conv(x)
        # x = Softmax(dim=1)(x)
        return x

def main():
    model = UNet(num_classes=59)
    x = paddle.rand(shape=(1,3,512,512), dtype='float32')
    pred = model(x)
    print(pred.shape)

if __name__ == '__main__':
    main()