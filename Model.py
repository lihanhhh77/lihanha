import torch
import torch.nn as nn
from utils import *
class BaseModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.input_width = configs['width']
        self.input_channels = 150
        self.channelNum = configs['channelNum']

        self.block1 = self.create_block(15)
        self.block2 = self.create_block(75)
        self.block3 = self.create_block(55)
        self.fusion = Concat()

        in_size = self.input_channels * 3
        self.Sconv3 = nn.Sequential(PointwiseConv2d(in_size, 100))
        self.log_layer1 = LogmLayer(100, vectorize=False)
        self.PCOM = PCOM(100)
        # self.log_layer1 = LogmLayer(100, vectorize=False)
        self.affine_layer = AffineInvariantLayer(100)  # Affine-Invariant分支
        self.vec = Vec(100)
        self.feature_pool = nn.AdaptiveAvgPool2d((None, 1))  # 只对最后一个维度(时间)池化
        self.projection = nn.Sequential(
            nn.Linear(100, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64)
        )

        self.FC = nn.Sequential(nn.Linear(5050, configs['class_num']))

        self.apply(self.initParms)

    def create_block(self, kernel_size):
        """
        创建卷积块
        :param kernel_size: 卷积核大小
        :return: 卷积块
        """
        return nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1), stride=1, padding='same', groups=1,
                      bias=False),
            # 逐点卷积：1个输入通道→300个输出通道（与原数量一致），使用1×1卷积核
            nn.Conv2d(in_channels=1, out_channels=300, kernel_size=(1, 1), stride=1, padding='same', bias=False),

            Conv2dWithConstraint(300, self.input_width, kernel_size=(self.channelNum, 1), padding=0, bias=False,
                                 groups=300),
            PointwiseConv2d(self.input_width, self.input_channels),
            LayerNormalization(1000),


            nn.Conv2d(self.input_channels, self.input_channels, kernel_size=(1, kernel_size), padding='same',
                      bias=False, groups=self.input_channels),
            LayerNormalization(1000),

        )

    def initParms(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x, return_projection=False):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        h1 = self.block1(x)
        h2 = self.block2(x)
        h3 = self.block3(x)
        h = self.fusion([h1, h2, h3])

        h = self.Sconv3(h)
        # print('h',h.shape)
        # 若为预训练且需要投影
        if return_projection:
            # 修正：使用2D池化处理4维张量，保留通道维度，压缩时间维度
            h_pooled = self.feature_pool(h)  # [batch, 100, 1, 1]
            # 展平为 [batch, 100]
            h_flat = h_pooled.view(h_pooled.size(0), -1)
            return self.projection(h_flat)
        # 微调阶段，正常走后续注意力等层
        # h = self.attention_module(h)
        print('h1',h.shape)
        h=self.PCOM(h)
        print('h2', h.shape)
        affine_inv = self.affine_layer(h)
        feature = self.log_layer1(affine_inv)
        flat_feature = self.vec(feature)
        return self.FC(flat_feature), flat_feature