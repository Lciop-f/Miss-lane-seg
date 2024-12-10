import paddle
from paddle import nn
from Semantic_Branch import Semantic_Branch
from Spatial_Branch import Spatial_Branch


class semantic_seg(nn.Layer):
    def __init__(self,inter_dim=320,num_classes=2):
        super().__init__()
        self.conv_seg = nn.Conv2D(128, num_classes, kernel_size=1)
        self.upscale = nn.Sequential(
            nn.BatchNorm2D(320),
            nn.ReLU(),
            nn.Conv2DTranspose(inter_dim,inter_dim,4,2,1),
            nn.BatchNorm2D(inter_dim),
            nn.ReLU(),
            nn.Conv2DTranspose(inter_dim,256,4,4),
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2D(256)
        self.bn2 = nn.BatchNorm2D(128)
        self.dropout = nn.Dropout(0.1)
        self.conv = nn.Conv2D(256,128,3,1,1)
        self.semantic_branch = Semantic_Branch()

    def forward(self, x):
        x = self.semantic_branch(x)
        x = self.upscale(x)
        x = self.conv(self.relu(self.bn1(x)))
        x = self.dropout(x)
        out = self.conv_seg(self.relu(self.bn2(x)))
        return out


class spatial_seg(nn.Layer):
    def __init__(self,in_dim,num_classes=2):
        super().__init__()
        self.conv_seg = nn.Conv2D(128, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2D(in_dim)
        self.bn2 = nn.BatchNorm2D(128)
        self.dropout = nn.Dropout(0.1)
        self.conv = nn.Conv2D(in_dim,128,3,1,1)

    def forward(self, x):
        x = self.conv(self.relu(self.bn1(x)))
        x = self.dropout(x)
        out = self.conv_seg(self.relu(self.bn2(x)))
        return out

class Decoder(nn.Layer):
    def __init__(self,
                 mode,
                 in_dim=384,
                num_classes=2,
                 ):
        super().__init__()
        self.sem_seg = semantic_seg(num_classes=num_classes)
        self.spa_seg = spatial_seg(in_dim=in_dim,num_classes=num_classes)
        self.upsample = nn.Upsample((512,512),mode="bilinear")
        self.mode = mode

    def forward_train(self, inputs):
        x = paddle.concat(inputs,axis=1)
        out1 = self.spa_seg(self.upsample(x))
        return out1

    def forward_eval(self,inputs):
        x = paddle.concat(inputs, axis=1)
        out2 = self.sem_seg(x)
        return out2

    def forward(self,inputs):
        if self.mode == "train_s":
            out = self.forward_train(inputs)
        elif self.mode == "eval_m":
            out = self.forward_eval(inputs)
        else:
            raise ValueError
        return out

class sctsea(nn.Layer):
    def __init__(self,mode="train_s"):
        super().__init__()
        self.backbone = Spatial_Branch()
        self.decoder = Decoder(mode=mode)
    def forward(self, x):
        return self.decoder(self.backbone(x))