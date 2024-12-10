import paddle
from paddle import nn
from Seaformer import Block

class Sea_Block(nn.Layer):
    def __init__(self, in_dim, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU):
        super().__init__()
        self.base_block = Block(dim, key_dim, num_heads, mlp_ratio=mlp_ratio, attn_ratio=attn_ratio, drop=drop,
                                drop_path=drop_path, act_layer=act_layer)
        self.conv1 = nn.Conv2D(in_dim, dim, 3, 2, 1)
        self.conv2 = nn.Conv2D(dim, dim, 1)
        self.conv2_1 = nn.Conv2D(dim, dim, 1)
        self.conv3 = nn.Conv2D(dim, dim, 3, 1, 1)
        self.bn1 = nn.BatchNorm2D(dim)
        self.act = act_layer()

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn1(self.conv2(x)))
        x = self.base_block(x)
        x = self.base_block(x)
        x = self.act(self.bn1(self.conv2_1(x)))
        x = self.act(self.bn1(self.conv3(x)))
        return x


class Semantic_Branch(nn.Layer):
    def __init__(self, in_dim=384, embed_dim=[128, 256, 320], key_dim=[16, 20, 24], num_heads=8):
        super().__init__()
        self.layers = nn.LayerList()
        self.layers.append(Sea_Block(in_dim, embed_dim[0], key_dim[0], num_heads))
        for i in range(1, len(embed_dim)):
            self.layers.append(nn.Sequential(
                nn.Upsample((64*i,64*i),mode="bilinear"),
                Sea_Block(embed_dim[i - 1], embed_dim[i], key_dim[i], num_heads)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x