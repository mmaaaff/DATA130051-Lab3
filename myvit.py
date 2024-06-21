import torch
import torch.nn as nn

# 定义Multi-Head Attention模块
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 定义线性层用于q, k, v
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        N, seq_length, d_model = x.shape
        assert d_model == self.d_model

        # 分别计算q, k, v
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 将q, k, v reshape并进行分组
        q = q.view(N, seq_length, self.num_heads, self.head_dim)
        k = k.view(N, seq_length, self.num_heads, self.head_dim)
        v = v.view(N, seq_length, self.num_heads, self.head_dim)

        # 转置操作使得头部维度放在batch前面
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算Attention分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5
        attention = torch.nn.functional.softmax(scores, dim=-1)

        # 使用Attention分数对v进行加权求和
        out = torch.matmul(attention, v)

        # 将头部维度还原
        out = out.transpose(1, 2).contiguous()
        out = out.view(N, seq_length, self.d_model)

        # 通过最终线性层
        out = self.fc_out(out)
        return out

# 定义Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 定义前馈神经网络
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 计算多头注意力
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))

        # 计算前馈神经网络
        mlp_out = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp_out))

        return x

# 定义Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, d_model, num_heads, mlp_dim, num_layers, num_classes):
        super(VisionTransformer, self).__init__()
        assert img_size % patch_size == 0

        self.img_size = img_size
        self.patch_size = patch_size

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size

        # 定义线性层用于Patch Embedding
        self.patch_embed = nn.Linear(self.patch_dim, d_model)

        # 定义位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.dropout = nn.Dropout(0.1)

        # 堆叠多个Transformer Block
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_dim) for _ in range(num_layers)
        ])

        # 定义最终分类层
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        N, C, H, W = x.shape
        assert H == W == self.img_size

        # 将图片拆分成Patch
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(N, self.num_patches, -1)

        # 进行Patch Embedding
        x = self.patch_embed(patches)

        # 加入分类Token
        cls_tokens = self.cls_token.expand(N, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        x = self.dropout(x)

        # 通过多个Transformer Block
        for transformer in self.transformer_blocks:
            x = transformer(x)

        # 取出分类Token并进行分类
        cls_token_final = x[:, 0]
        out = self.mlp_head(cls_token_final)
        return out

# 测试Vision Transformer
vit = VisionTransformer(img_size=32, patch_size=4, d_model=128, num_heads=16, mlp_dim=256, num_layers=4, num_classes=100)
dummy_input = torch.randn(4, 3, 32, 32)
out = vit(dummy_input)
print(out.shape)  # 输出为 (batchsize, 100)
