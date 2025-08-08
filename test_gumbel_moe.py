from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.utils.data
import json
import numpy as np
from progress.bar import Bar
from copy import deepcopy
import time
import matplotlib

matplotlib.use('Agg')  # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt

# --- Distributed Imports ---
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap

# --- Project-specific Imports ---
from lib.utils.opts import opts
from lib.models.Gumbel_MoE_DSFNet import Gumbel_MoE_DSFNet
from lib.models.DSFNet import DSFNet as DSFNet_expert
from lib.dataset.coco import COCO as CustomDataset
from lib.utils.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process
from detection_evaluator import DetectionEvaluator


# ==================== [Helper Functions - No Changes Needed] ====================

def process(model, image, opt):
    """Model inference function"""
    with torch.no_grad():
        model_return = model(image)
        output = model_return[0][-1] if isinstance(model_return, tuple) else model_return[-1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output.get('reg', None)
        dets = ctdet_decode(hm, wh, reg=reg, K=opt.K)
    return dets


def post_process(dets, meta, num_classes):
    """Post-processing to convert heatmap coordinates to original image coordinates"""
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        if len(dets[0][j]) > 0:
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        else:
            dets[0][j] = np.empty([0, 5], dtype=np.float32)
    return dets[0]


def save_predictions_as_yolo(predictions, original_img_shape, save_path, coco_id_to_yolo_id_map):
    """Saves detection results to a YOLO .txt format file."""
    img_h, img_w = original_img_shape
    with open(save_path, 'w') as f:
        for coco_cls_id in predictions:
            yolo_cls_id = coco_id_to_yolo_id_map.get(coco_cls_id)
            if yolo_cls_id is None:
                continue
            for bbox in predictions[coco_cls_id]:
                score = bbox[4]
                x1, y1, x2, y2 = bbox[:4]
                x1, x2 = np.clip([x1, x2], 0, img_w - 1)
                y1, y2 = np.clip([y1, y2], 0, img_h - 1)
                box_w, box_h = x2 - x1, y2 - y1
                if box_w <= 0 or box_h <= 0: continue
                center_x, center_y = x1 + box_w / 2, y1 + box_h / 2
                f.write(
                    f"{yolo_cls_id} {center_x / img_w:.6f} {center_y / img_h:.6f} {box_w / img_w:.6f} {box_h / img_h:.6f} {score:.6f}\n")


def save_and_plot_results(results, model_name, model_path, opt):
    # This function remains unchanged from the previous version.
    model_file_name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join(opt.exp_dir, f'confidence_analysis_{model_file_name}')
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'confidence_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📁 Full evaluation results saved to: {json_path}")
    thresholds = sorted([float(k) for k in results.keys()])
    if len(thresholds) < 1: return
    recalls = [results[str(t)]['recall'] for t in thresholds]
    fars = [results[str(t)]['false_alarm_rate'] for t in thresholds]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Confidence Threshold');
    ax1.set_ylabel('Recall', color=color)
    ax1.plot(thresholds, recalls, marker='o', color=color, label='Recall')
    ax1.tick_params(axis='y', labelcolor=color);
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_ylim([0, max(1.0, max(recalls) * 1.1 if recalls else 1.0)])
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('False Alarm Rate', color=color)
    ax2.plot(thresholds, fars, marker='s', linestyle='--', color=color, label='False Alarm Rate')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, max(1.0, max(fars) * 1.1 if fars else 1.0)])
    fig.suptitle('Recall & False Alarm Rate vs. Confidence Threshold', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    lines, labels = ax1.get_legend_handles_labels();
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    plot_path = os.path.join(output_dir, 'Recall_FAR_vs_Confidence.png')
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"📈 Performance curves saved to: {plot_path}")


# ==================== [Distributed Setup & Main Worker] ====================

def setup(opt):
    """Initializes the distributed process group based on environment variables."""
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group("nccl")  # `backend` can be omitted, NCCL is default for GPUs
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    opt.local_rank = int(os.environ['LOCAL_RANK'])
    return rank, world_size


def gather_files(pred_root_dir, world_size, rank):
    """Gathers prediction files from all ranks to rank 0."""
    if world_size == 1: return
    dist.barrier()
    if rank == 0:
        print("\nAggregating prediction files from all ranks...")
        for r in range(1, world_size):
            rank_dir = f"{pred_root_dir}_rank_{r}"
            if os.path.exists(rank_dir):
                for video_name in os.listdir(rank_dir):
                    src_video_dir = os.path.join(rank_dir, video_name)
                    dst_video_dir = os.path.join(pred_root_dir, video_name)
                    os.makedirs(dst_video_dir, exist_ok=True)
                    for fname in os.listdir(src_video_dir):
                        os.rename(os.path.join(src_video_dir, fname), os.path.join(dst_video_dir, fname))
                try:
                    os.rmdir(rank_dir)
                except OSError as e:
                    print(f"Could not remove temp directory {rank_dir}: {e}")
    dist.barrier()


def test_main(opt):
    """Main function to be run by each process."""
    rank, world_size = setup(opt)

    dataset = CustomDataset(opt, 'test')
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, sampler=sampler
    )

    head = {'hm': dataset.num_classes, 'wh': 2, 'reg': 2}
    head_conv = 128
    num_experts_target = 15
    # 在这里定义你想要的 top_k 值 (应与训练时使用的值匹配)
    TOP_K_VALUE = 3  # <<<<<<<<<<<<<<<<<<<< 在这里修改激活的专家数量

    # 仅为结构，不加载权重
    expert_list = torch.nn.ModuleList([DSFNet_expert(head, head_conv) for _ in range(num_experts_target)])

    model = Gumbel_MoE_DSFNet(
        heads=head,
        head_conv=head_conv,
        num_experts=num_experts_target,
        top_k=TOP_K_VALUE,  # <<<<<<<<<<<<<<<<<<<< 将参数传递给模型
        expert_modules=expert_list
    ).to(rank)

    print(f"[Rank {rank}] Manually wrapping {len(model.experts)} expert modules for FSDP...")
    for i, expert_layer in enumerate(model.experts):
        model.experts[i] = wrap(expert_layer)

    fsdp_model = FSDP(model)

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

    if rank == 0:
        print("✅ Trained MoE model loaded on all ranks.")

    dist.barrier()
    fsdp_model.eval()

    coco_id_to_yolo_id = dataset.cat_ids

    pred_root_dir = os.path.join(opt.save_results_dir, 'predictions_raw')
    rank_pred_dir = f"{pred_root_dir}_rank_{rank}"
    os.makedirs(rank_pred_dir, exist_ok=True)

    if rank == 0:
        bar = Bar('🚀 Inference Phase', max=len(data_loader))

    for ind, (img_id, batch) in enumerate(data_loader):
        image = batch['input'].to(rank)
        meta = {k: v.numpy()[0] for k, v in batch['meta'].items()}
        original_h, original_w = meta['original_height'], meta['original_width']
        file_rel_path = dataset.coco.loadImgs(ids=[img_id.item()])[0]['file_name']

        dets_raw = process(fsdp_model, image, opt)
        dets_processed = post_process(dets_raw, meta, dataset.num_classes)

        path_parts = file_rel_path.replace('\\', '/').split('/')
        video_name = path_parts[-2] if len(path_parts) > 1 else 'video_root'
        frame_name_no_ext = os.path.splitext(os.path.basename(file_rel_path))[0]
        save_video_dir = os.path.join(rank_pred_dir, video_name)
        os.makedirs(save_video_dir, exist_ok=True)
        save_path = os.path.join(save_video_dir, frame_name_no_ext + '.txt')

        save_predictions_as_yolo(dets_processed, (meta['out_height'], meta['out_width']), (original_h, original_w),
                                 save_path, coco_id_to_yolo_id)

        if rank == 0:
            bar.next()

    if rank == 0: bar.finish()

    gather_files(pred_root_dir, world_size, rank)
    dist.destroy_process_group()


# ==================== [Main Execution Block] ====================
if __name__ == '__main__':
    opts_parser = opts()
    opt = opts_parser.parse()
    opt.model_name = 'Gumbel_MoE_DSFNet'
    opt = opts_parser.init(opt)

    if opt.load_model == '':
        print("❌ Error: Please specify model path with --load_model")
        exit()

    # The main logic will be run by torchrun
    test_main(opt)

    # --- Evaluation now happens strictly on rank 0 after all processes finish ---
    if int(os.environ.get('RANK', '0')) == 0:
        results_summary = {}
        confidence_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

        pred_root_dir_final = os.path.join(opt.save_results_dir, 'predictions_raw')

        print("\n📊 Starting multi-confidence evaluation on aggregated results...")
        for i, conf in enumerate(confidence_thresholds):
            print(f"🎯 Evaluating confidence threshold {i + 1}/{len(confidence_thresholds)}: {conf:.3f}")
            filtered_pred_root_dir = os.path.join(opt.save_results_dir, f'filtered_preds_conf_{conf:.3f}')
            os.makedirs(filtered_pred_root_dir, exist_ok=True)

            for video_name in os.listdir(pred_root_dir_final):
                src_video_dir = os.path.join(pred_root_dir_final, video_name)
                if not os.path.isdir(src_video_dir): continue
                dst_video_dir = os.path.join(filtered_pred_root_dir, video_name)
                os.makedirs(dst_video_dir, exist_ok=True)
                for fname in os.listdir(src_video_dir):
                    src_file_path = os.path.join(src_video_dir, fname)
                    if not os.path.isfile(src_file_path): continue
                    with open(src_file_path, 'r') as f_in, open(os.path.join(dst_video_dir, fname), 'w') as f_out:
                        for line in f_in:
                            try:
                                if float(line.strip().split()[-1]) >= conf: f_out.write(line)
                            except (ValueError, IndexError):
                                continue

            eval_config = {
                'gt_root': os.path.join(opt.data_dir, 'labels'),
                'pred_root': filtered_pred_root_dir,
                'iou_threshold': opt.iou_thresh,
                'class_names': CustomDataset(opt, 'test').class_name[1:],
            }
            evaluator = DetectionEvaluator(eval_config)
            evaluator.evaluate_all()
            overall_metrics = evaluator.calculate_overall_metrics()

            if 'overall' in overall_metrics:
                metrics = overall_metrics['overall']
                results_summary[str(conf)] = {
                    'recall': metrics.get('recall', 0.0), 'precision': metrics.get('precision', 0.0),
                    'f1': metrics.get('f1', 0.0), 'false_alarm_rate': metrics.get('false_alarm_rate', 1.0),
                    'spatiotemporal_stability': metrics.get('spatiotemporal_stability', 0.0),
                    'tp': metrics.get('tp', 0), 'fp': metrics.get('fp', 0), 'fn': metrics.get('fn', 0)
                }
            else:
                results_summary[str(conf)] = {'recall': 0, 'precision': 0, 'f1': 0, 'false_alarm_rate': 1.0,
                                              'spatiotemporal_stability': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}

        print("✅ Evaluation complete!")

        if results_summary:
            print("\n" + "=" * 95)
            print("📊 Confidence Threshold Performance Summary")
            print("=" * 95)
            print(
                f"{'Conf.':<8} {'Recall':<10} {'Precision':<10} {'FAR':<10} {'Stability':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
            print("-" * 95)
            for conf, metrics in sorted(results_summary.items(), key=lambda item: float(item[0])):
                print(f"{float(conf):<8.3f} {metrics['recall']:<10.4f} {metrics['precision']:<10.4f} "
                      f"{metrics['false_alarm_rate']:<10.4f} {metrics['spatiotemporal_stability']:<12.4f} "
                      f"{metrics['tp']:<8} {metrics['fp']:<8} {metrics['fn']:<8}")
            print("=" * 95)
            save_and_plot_results(results_summary, opt.model_name, opt.load_model, opt)

        print("\n✅ Multi-confidence evaluation finished!")