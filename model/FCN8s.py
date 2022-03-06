
import paddle
from paddle.nn import Conv2D, MaxPool2D ,Conv2DTranspose ,BatchNorm2D, ReLU, Pad2D

class Conv_bn_relu(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, relu=True):
        super(Conv_bn_relu, self).__init__()
        self.conv = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding)
        self.bn = BatchNorm2D(num_features=out_channels)
        if relu:
            self.relu = ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FCN8s(paddle.nn.Layer):
    def __init__(self, num_classes=19):
        super(FCN8s, self).__init__()
        self.conv1_1 = Conv_bn_relu(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv1_2 = Conv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPool2D(kernel_size=2, padding=1)  # 1/2

        self.conv2_1 = Conv_bn_relu(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = Conv_bn_relu(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = MaxPool2D(kernel_size=2, padding=1)  # 1/4

        self.conv3_1 = Conv_bn_relu(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = Conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = Conv_bn_relu(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = MaxPool2D(kernel_size=2, padding=1)  # 1/8

        self.conv4_1 = Conv_bn_relu(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = Conv_bn_relu(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = Conv_bn_relu(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool4 = MaxPool2D(kernel_size=2, padding=1)  # 1/16

        self.conv5_1 = Conv_bn_relu(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = Conv_bn_relu(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = Conv_bn_relu(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool5 = MaxPool2D(kernel_size=2, padding=1)  # 1/32

        self.conv6 = Conv_bn_relu(in_channels=512, out_channels=4096, kernel_size=7)
        self.conv7 = Conv_bn_relu(in_channels=4096, out_channels=4096, kernel_size=1)

        self.score = Conv2D(in_channels=4096, out_channels=num_classes, kernel_size=1)
        self.score_pool3 = Conv2D(in_channels=256, out_channels=num_classes, kernel_size=1)
        self.score_pool4 = Conv2D(in_channels=512, out_channels=num_classes, kernel_size=1)

        self.up_score = Conv2DTranspose(in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2)
        self.up_pool4 = Conv2DTranspose(in_channels=num_classes, out_channels=num_classes, kernel_size=4, stride=2)
        self.up_pool3 = Conv2DTranspose(in_channels=num_classes, out_channels=num_classes, kernel_size=16, stride=8)


    def forward(self, input):
        x = Pad2D([100, 100, 100, 100])(input)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        # print(x.shape)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        # print(x.shape)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        # print(x.shape)
        x3 = self.score_pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        # print(x.shape)
        x4 = self.score_pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        # print(x.shape)
        x = self.conv6(x)
        x = self.conv7(x)


        x = self.score(x)
        x = self.up_score(x)
        crop4_h = (x4.shape[2] - x.shape[2])//2
        crop4_w = (x4.shape[3] - x.shape[3])//2
        x4 = x4[:, : ,crop4_h:crop4_h +x.shape[2], crop4_w:crop4_w +x.shape[3]]
        x = x + x4

        x = self.up_pool4(x)
        crop3_h = (x3.shape[2] - x.shape[2])//2
        crop3_w = (x3.shape[3] - x.shape[3])//2
        x3 = x3[:, : ,crop3_h:crop3_h +x.shape[2], crop3_w:crop3_w +x.shape[3]]
        x = x + x3

        x = self.up_pool3(x)
        crop_h = (x.shape[2] - input.shape[2])//2
        crop_w = (x.shape[3] - input.shape[3])//2
        x = x[:, : ,crop_h:crop_h +input.shape[2], crop_w:crop_w +input.shape[3]]
        return x

def main():
    model = FCN8s()
    data = paddle.rand((1 ,3 ,512 ,512))
    # print(data)
    y = model(data)
    print(y.shape)


if __name__ == '__main__':
    main()