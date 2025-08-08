from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.utils.data
from copy import deepcopy

# --- Distributed Training Imports ---
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap

# --- Project-specific Imports ---
from lib.utils.opts import opts
from lib.utils.logger import Logger
from lib.models.stNet import save_model
from lib.models.DSFNet import DSFNet as DSFNet_expert
from lib.models.Gumbel_MoE_DSFNet import Gumbel_MoE_DSFNet
from lib.dataset.coco_rsdata import COCO
from lib.Trainer.ctdet import CtdetTrainer


def perturb_model_weights(model, noise_level=1e-4):
    """
    对模型的所有可训练权重和偏置添加一个小的随机高斯噪声。
    """
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                noise = torch.randn_like(param) * noise_level
                param.add_(noise)
    return model


def setup(opt):
    """
    使用由 torchrun 提供的环境变量初始化分布式进程组。
    """
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    opt.local_rank = local_rank
    return rank, world_size


def main(opt):
    """
    主训练函数，由每个进程独立执行。
    """
    rank, world_size = setup(opt)
    logger = Logger(opt) if rank == 0 else None

    torch.manual_seed(opt.seed + rank)

    DataTrain = COCO(opt, 'train')
    train_sampler = DistributedSampler(DataTrain, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(
        DataTrain, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler
    )

    # 验证集只在主进程上加载和使用
    val_loader = None
    if rank == 0:
        DataVal = COCO(opt, 'test')
        val_loader = torch.utils.data.DataLoader(
            DataVal, batch_size=1, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True
        )

    print(f"[Rank {rank}] Creating model...")
    head = {'hm': DataTrain.num_classes, 'wh': 2, 'reg': 2}
    head_conv = 128

    base_pretrained_paths = [
        './checkpoint/DSFNet_s1.pth',
        './checkpoint/DSFNet_s2.pth',
        './checkpoint/DSFNet_s3.pth',
    ]
    num_experts_target = 15
    replicas_per_pth = 4

    expert_list = torch.nn.ModuleList()
    for path in base_pretrained_paths:
        base_expert = DSFNet_expert(head, head_conv)
        if os.path.exists(path):
            from lib.models.stNet import load_model
            base_expert = load_model(base_expert, path)
        expert_list.append(base_expert)
        for i in range(replicas_per_pth):
            perturbed_expert = deepcopy(base_expert)
            perturbed_expert = perturb_model_weights(perturbed_expert)
            expert_list.append(perturbed_expert)

    # 在这里定义你想要的 top_k 值
    TOP_K_VALUE = 4 # <<<<<<<<<<<<<<<<<<<< 在这里修改激活的专家数量

    model = Gumbel_MoE_DSFNet(
        heads=head,
        head_conv=head_conv,
        num_experts=num_experts_target,
        top_k=TOP_K_VALUE, # <<<<<<<<<<<<<<<<<<<< 将参数传递给模型
        expert_modules=expert_list
    ).to(rank)

    # --- Manual Wrapping for PyTorch 1.11.0 ---
    print(f"[Rank {rank}] Manually wrapping {len(model.experts)} expert modules for FSDP...")
    for i, expert_layer in enumerate(model.experts):
        model.experts[i] = wrap(expert_layer)

    fsdp_model = FSDP(model)
    print(f"[Rank {rank}] Model wrapped with FSDP using manual wrapping.")

    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=opt.lr)

    start_epoch = 0
    if opt.load_model != '':
        if rank == 0:
            checkpoint = torch.load(opt.load_model, map_location='cpu')
        else:
            checkpoint = None
        dist.barrier()
        broadcast_obj = [checkpoint] if rank == 0 else [None]
        dist.broadcast_object_list(broadcast_obj, src=0)
        checkpoint = broadcast_obj[0]

        with FSDP.summon_full_params(fsdp_model, writeback=True, rank0_only=False):
            fsdp_model.load_state_dict(checkpoint['state_dict'])
            if opt.resume and 'optimizer' in checkpoint:
                start_epoch = checkpoint['epoch']
                print(f"Resuming from epoch {start_epoch}. Optimizer state not loaded.")
    dist.barrier()

    trainer = CtdetTrainer(opt, fsdp_model, optimizer)
    trainer.set_device(opt.gpus, rank)

    print(f'[Rank {rank}] Starting training...')
    best = -1
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        train_sampler.set_epoch(epoch)  # Important for shuffling
        log_dict_train, _ = trainer.train(epoch, train_loader)

        if rank == 0:
            logger.write(f'epoch: {epoch} |')
            save_model(os.path.join(opt.save_weights_dir, 'model_last.pth'), epoch, fsdp_model, optimizer)
            for k, v in log_dict_train.items():
                logger.write(f'{k} {v:8f} | ')

            if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
                with torch.no_grad():
                    log_dict_val, _, stats = trainer.val(epoch, val_loader, DataVal.coco, DataVal)
                for k, v in log_dict_val.items():
                    logger.write(f'{k} {v:8f} | ')
                if log_dict_val['ap50'] > best:
                    best = log_dict_val['ap50']
                    save_model(os.path.join(opt.save_weights_dir, 'model_best.pth'), epoch, fsdp_model)

            logger.write('\n')

        if epoch in opt.lr_step:
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            if rank == 0:
                print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    if rank == 0:
        logger.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    opts_parser = opts()
    opt = opts_parser.parse()
    opt.model_name = 'Gumbel_MoE_DSFNet_15_Experts'
    opt = opts_parser.init(opt)

    # torchrun will handle launching multiple processes, each running this script.
    main(opt)