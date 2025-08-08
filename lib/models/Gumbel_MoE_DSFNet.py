import torch
from torch import nn
import torch.nn.functional as F
import os

# 导入原始的DSFNet作为我们的“专家”
from .DSFNet import DSFNet as DSFNet_expert, fill_fc_weights
# 导入您项目中的 load_model 函数
from .stNet import load_model


class GumbelGatingNetwork(nn.Module):
    """
    使用Gumbel-Softmax的门控网络，支持Top-K专家选择。
    - 训练时使用 Gumbel-Softmax 实现对Top-K个专家的可导选择。
    - 评估时使用 ArgMax 实现对Top-K个专家的确定性选择。
    """

    def __init__(self, num_experts, top_k=1):
        """
        初始化门控网络。
        Args:
            num_experts (int): 专家总数。
            top_k (int): 每个输入需要激活的专家数量。
        """
        super(GumbelGatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 一个轻量级的CNN，用于从输入中提取特征以决定专家权重
        self.backbone = nn.Sequential(
            # 输入形状: [B, C*T, H, W], 例如 [B, 15, H, W]
            nn.Conv2d(3 * 5, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 输出特征图尺寸 H/4, W/4
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, self.num_experts)

    def forward(self, x):
        b, c, t, h, w = x.shape
        # 将时间和通道维度合并，以适应2D卷积: [B, C, T, H, W] -> [B, C*T, H, W]
        x_reshaped = x.view(b, c * t, h, w)

        features = self.backbone(x_reshaped)
        features = self.avg_pool(features)
        features = features.view(b, -1)

        # logits: [B, num_experts], 每个专家的原始分数
        logits = self.fc(features)

        # 选出 Top-K 的专家索引和对应的分数(logits)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=1)

        # 创建一个用于存储门控权重的稀疏张量
        sparse_gates = torch.zeros_like(logits)

        if self.training:
            # 训练时: 在 top-k 的 logits 上应用 Gumbel-Softmax
            # hard=True 会在前向传播时产生 one-hot 输出，但在反向传播时使用 softmax 的梯度
            gumbel_top_k = F.gumbel_softmax(top_k_logits, tau=1.0, hard=True, dim=-1)

            # 将 Gumbel-Softmax 的结果（近似0/1）放回稀疏张量的正确位置
            sparse_gates.scatter_(1, top_k_indices, gumbel_top_k)
        else:
            # 评估时: 直接将被选中的专家位置赋予权重
            # 这里我们使用 1.0 / self.top_k，实现对 top-k 个专家的等权重平均集成
            sparse_gates.scatter_(1, top_k_indices, 1.0 / self.top_k)

        # 辅助损失 (Load Balancing Loss)，鼓励门控网络均匀地使用所有专家
        # 1. 创建一个真正的0/1掩码来表示哪些专家被选中
        expert_mask = torch.zeros_like(logits).scatter_(1, top_k_indices, 1)
        # 2. 计算每个专家在批次中被选中的频率
        density = expert_mask.mean(dim=0)
        # 3. 计算每个专家的平均路由概率
        density_proxy = F.softmax(logits, dim=1).mean(dim=0)
        # 4. 损失是这两者的点积之和，再乘以专家数量以保持尺度
        aux_loss = (density * density_proxy).sum() * self.num_experts

        return sparse_gates, top_k_indices, aux_loss


class Gumbel_MoE_DSFNet(nn.Module):
    """
    使用Gumbel-Softmax门控进行端到端训练的MoE模型。
    支持选择 Top-K 个专家。
    """

    def __init__(self, heads, head_conv=128, num_experts=3, top_k=1, loss_coef=1e-2,
                 pretrained_paths=None, expert_modules=None):
        super(Gumbel_MoE_DSFNet, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.loss_coef = loss_coef
        self.heads = heads

        print(f"🚀 初始化 Gumbel-MoE-DSFNet 模型 (Top-K={self.top_k})")
        print(f"   - 门控机制: 训练时 Top-K Gumbel-Softmax, 评估时 Top-K ArgMax")
        print(f"   - 专家总数: {self.num_experts}")

        # 将 top_k 参数传递给门控网络
        self.gating_network = GumbelGatingNetwork(self.num_experts, self.top_k)

        if expert_modules is not None:
            print("   - 使用外部提供的专家模块列表。")
            if len(expert_modules) != self.num_experts:
                raise ValueError(f"提供的专家模块数量({len(expert_modules)})与num_experts({self.num_experts})不匹配。")
            self.experts = expert_modules
        elif pretrained_paths is not None:
            print("   - 根据路径列表创建并初始化专家。")
            if len(pretrained_paths) * (1 + 4) != self.num_experts and len(
                    pretrained_paths) != self.num_experts:  # 兼容旧的初始化逻辑
                raise ValueError(f"提供的路径数量({len(pretrained_paths)})与专家数量({self.num_experts})配置不匹配。")

            self.experts = nn.ModuleList()
            for i, path in enumerate(pretrained_paths):
                print(f"   - 初始化专家 {i + 1}/{len(pretrained_paths)} (及副本) 从: '{os.path.basename(path)}'")
                expert_model = DSFNet_expert(heads, head_conv)
                if os.path.exists(path):
                    expert_model = load_model(expert_model, path)
                else:
                    print(f"   - ⚠️ 警告: 路径不存在 {path}。专家将使用随机初始化权重。")

                # 根据 expert_list 的长度来决定是否要复制专家
                if len(self.experts) < self.num_experts:
                    self.experts.append(expert_model)

                # 如果需要复制专家以达到 num_experts
                replicas_to_add = (self.num_experts // len(pretrained_paths)) - 1
                for _ in range(replicas_to_add):
                    if len(self.experts) < self.num_experts:
                        perturbed_expert = deepcopy(expert_model)
                        # 这里可以添加扰动逻辑，但为了简化，我们先直接复制
                        self.experts.append(perturbed_expert)

        else:
            raise ValueError("必须提供 'pretrained_paths' 或 'expert_modules'。")

        print(f"✅ 所有 {len(self.experts)} 个专家模块已设置。")

    def forward(self, x):
        # 1. 从门控网络获取稀疏权重
        # sparse_gates 形状: [batch_size, num_experts]
        # 值示例 (训练, k=2): [0, 1, 0, 0, 1, 0, ...]
        # 值示例 (测试, k=2): [0, 0.5, 0, 0, 0.5, 0, ...]
        sparse_gates, top_k_indices, aux_loss = self.gating_network(x)

        # 2. 计算所有专家的输出
        # 这是为了确保梯度能够流向所有专家，即使它们当前未被激活
        expert_outputs = [expert(x)[0] for expert in self.experts]

        # 3. 初始化一个字典来存储最终的聚合输出
        final_outputs = {head: 0.0 for head in self.heads}

        # 4. 加权求和
        for i in range(self.num_experts):
            # 获取第i个专家的门控权重, 形状: [batch_size]
            # 并将其 reshape 以便与输出张量进行广播: [batch_size, 1, 1, 1]
            gate_reshaped = sparse_gates[:, i].view(-1, 1, 1, 1)

            # 获取第i个专家的输出字典
            expert_head_dict = expert_outputs[i]

            # 遍历每个输出头 (e.g., 'hm', 'wh', 'reg')
            for head_name, head_tensor in expert_head_dict.items():
                # 将该专家的输出乘以其权重，并累加到最终结果中
                # 如果一个专家未被选中，其 gate_reshaped 的值为0，乘积也为0
                final_outputs[head_name] += gate_reshaped * head_tensor

        # 返回最终的预测结果和用于训练的辅助损失
        return [final_outputs], self.loss_coef * aux_loss