from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from thop import profile
from lib.models.DSFNet_with_Static import DSFNet_with_Static
from lib.models.DSFNet import DSFNet
from lib.models.DSFNet_with_Dynamic import DSFNet_with_Dynamic
# ==================== [核心修改] ====================
from lib.models.Sparse_Ensemble_MoE_DSFNet import SparseEnsembleMoE_DSFNet
# =======================================================
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def model_lib(model_chose):
    model_factory = {
                     'DSFNet_with_Static': DSFNet_with_Static,
                     'DSFNet': DSFNet,
                     'DSFNet_with_Dynamic': DSFNet_with_Dynamic,
                    # ==================== [核心修改] ====================
                     'SparseEnsembleMoE_DSFNet': SparseEnsembleMoE_DSFNet,
                    # =======================================================
                     }
    return model_factory[model_chose]

def get_det_net(heads, model_name):
    model_name = model_lib(model_name)
    model = model_name(heads)
    return model


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
    # FSDP的加载逻辑更复杂，最好在训练脚本中直接处理
    # 这个通用函数保留用于非FSDP模型
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # ... (其余检查逻辑保持不变) ...

    model.load_state_dict(state_dict, strict=False)

    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')

    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


# --- 找到 save_model 函数并替换其 FSDP 逻辑 ---
def save_model(path, epoch, model, optimizer=None):
    # 检查模型是否是FSDP实例
    if isinstance(model, FSDP):
        # PyTorch 1.11.0 的 FSDP 保存方式
        if torch.distributed.get_rank() == 0:
            print("Gathering full state dict for saving...")

        # 使用 summon_full_params 来重新组合完整的模型参数以进行保存
        with FSDP.summon_full_params(model, writeback=False, rank0_only=True):
            # 只有 rank 0 会接收到完整的、在 CPU 上的 state_dict
            state_dict = model.state_dict()
            if torch.distributed.get_rank() == 0:
                data = {'epoch': epoch, 'state_dict': state_dict}
                if optimizer is not None:
                    # 注意: 保存优化器状态在 FSDP 中比较复杂，这里暂时跳过
                    # data['optimizer'] = optimizer.state_dict()
                    pass
                torch.save(data, path)
                print(f"Full state dict saved by rank 0 to {path}")

    else:  # 非分布式或普通DDP
        print(f"wrong1")
