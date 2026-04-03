
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import argparse
import datetime
import json
import math
import os
import time
from collections import OrderedDict
from pathlib import Path

import deepspeed
from geoopt import ManifoldParameter
from geoopt.optim import RiemannianAdam
from timm.optim import create_optimizer

import utils as utils
from config import train_label_paths, dev_label_paths, test_label_paths
from datasets import S2T_Dataset
from models.loop import Sign_Loop_Hyperbolic
from SLRT_metrics import islr_performance

def rank0_print(*args):
    if utils.is_main_process():
        print(*args)

def build_dataloaders(args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    rank0_print(f"Creating datasets...")
    
    train_data = S2T_Dataset(path=train_label_paths[args.dataset], args=args, phase='train')    

    if utils.get_world_size() > 1:
        train_sampler = DistributedSampler(
            train_data, 
            num_replicas=utils.get_world_size(), 
            rank=utils.get_rank(), 
            shuffle=True
        )
    else:
        train_sampler = SequentialSampler(train_data)
        
    train_dataloader = DataLoader(
            train_data,
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            collate_fn=train_data.collate_fn,
            sampler=train_sampler,   
            shuffle=False,           
            pin_memory=args.pin_mem,
            drop_last=True
        )

    this_phase = 'val' if 'MSASL' in args.dataset else 'dev'
    dev_data = S2T_Dataset(path=dev_label_paths[args.dataset], args=args, phase=this_phase)
    test_data = S2T_Dataset(path=test_label_paths[args.dataset], args=args, phase='test')

    if world_size > 1:
        dev_sampler = DistributedSampler(dev_data, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        dev_sampler = SequentialSampler(dev_data)
        test_sampler = SequentialSampler(test_data)

    dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size, sampler=dev_sampler, 
                                num_workers=args.num_workers, collate_fn=dev_data.collate_fn)
    
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, sampler=test_sampler, 
                                 num_workers=args.num_workers, collate_fn=test_data.collate_fn)

    if args.debug:
        indices = list(train_sampler)
        print(f"Rank {utils.get_rank()} first 5 indices: {indices[:5]}")
        dev_indices = list(dev_sampler)
        print(f"Rank {utils.get_rank()} DEV first 5 indices: {dev_indices[:5]}")
        test_indices = list(test_sampler)
        print(f"Rank {utils.get_rank()} TEST first 5 indices: {test_indices[:5]}")

    return train_dataloader, dev_dataloader, test_dataloader, train_sampler, dev_sampler, test_sampler

def build_model(args, device):
    rank0_print(f"Creating model: {args.loop}")
    
    model = Sign_Loop_Hyperbolic(args=args)

    if not args.use_deepspeed:
        model.to(device)

    for _, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    return model

def load_checkpoint(model, path, load_mode='finetune'):
    rank0_print('***********************************')
    rank0_print(f'Load Checkpoint {path} ({load_mode}) ...')
    rank0_print('***********************************')
    
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_k] = v

    ret = model.load_state_dict(new_state_dict, strict=False)
    
    rank0_print('Missing keys: \n', '\n'.join(ret.missing_keys))
    rank0_print(f'Total missing keys: {len(ret.missing_keys)}')
    rank0_print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))
    rank0_print(f'Total unexpected keys: {len(ret.unexpected_keys)}')
    
    start_epoch = 0
    if load_mode == 'resume' and 'epoch' in checkpoint:
        if 'checkpoint' in path:
            try:
                start_epoch = int(path.split('/')[-1].split('.')[0].split('_')[-1]) + 1
            except:
                start_epoch = checkpoint.get('epoch', 0)
    
    return start_epoch

def configure_optimizers(args, model_without_ddp, train_dataloader_len):
    hyp_optimizer = None
    optimizer = None

    if args.hyp_sep_lr:
        params = []
        hyp_params = []
        for name, p in model_without_ddp.named_parameters():
            if not p.requires_grad: continue
            if isinstance(p, ManifoldParameter) or name.endswith("manifold.c"):
                hyp_params.append(p)
            else:
                params.append(p)

        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=getattr(args, 'opt_betas', (0.9, 0.98))
        )
        
        if hyp_params:
            hyp_optimizer = RiemannianAdam(
                hyp_params,
                lr=args.hyp_lr,
                stabilize=getattr(args, 'hyp_stabilize', True),
                weight_decay=0.0
            )
            model_without_ddp.hyp_optimizer = hyp_optimizer
    else:
        optimizer = create_optimizer(args, model_without_ddp)

    return optimizer, hyp_optimizer

def get_deepspeed_config(args):
    config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clipping": args.gradient_clipping,
        "zero_optimization": {
            "stage": getattr(args, 'zero_stage', 3),
            "offload_optimizer": {
                "device": "cpu", "pin_memory": True
            } if getattr(args, 'offload', False) else None,
            "offload_param": {
                "device": "cpu", "pin_memory": True
            } if getattr(args, 'offload', False) else None,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "fp16": {
            "enabled": getattr(args, 'dtype', None) == 'fp16',
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": args.opt_betas,
                "eps": args.opt_eps,
                "weight_decay": args.weight_decay
            }
        }
    }
    if config["zero_optimization"]["offload_optimizer"] is None:
        del config["zero_optimization"]["offload_optimizer"]
    if config["zero_optimization"]["offload_param"] is None:
        del config["zero_optimization"]["offload_param"]
        
    return config

def main(args):
    rank0_print("Distributed mode with deepspeed:", args.use_deepspeed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank0_print("The device is:", device)
    args.device = device
    
    utils.set_seed(args.seed)

    if args.use_deepspeed:
        if not torch.distributed.is_initialized():
            deepspeed.init_distributed()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    train_dataloader, dev_dataloader, test_dataloader, train_sampler, dev_sampler, test_sampler = build_dataloaders(args)

    model = build_model(args, device)
    model_without_ddp = model

    if not args.use_deepspeed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    
    if args.finetune:
        load_checkpoint(model_without_ddp, args.finetune, load_mode='finetune')

    optimizer, hyp_optimizer = configure_optimizers(args, model_without_ddp, len(train_dataloader))
    
    if args.use_deepspeed:
        ds_config = get_deepspeed_config(args)
        rank0_print("Initializing DeepSpeed...")
        model, optimizer, _, _ = deepspeed.initialize(
            model=model_without_ddp, config=ds_config
        )
        model_without_ddp = model.module

    if args.resume:
        start_epoch = load_checkpoint(model_without_ddp, args.resume, load_mode='resume')
        args.start_epoch = start_epoch

    output_dir = Path(args.output_dir)
    rank0_print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        
        rank0_print('Training')
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(args, model, model_without_ddp, train_dataloader, optimizer, epoch, hyp_optimizer=hyp_optimizer)

        should_save = args.save_all_checkpoints or ((epoch + 1) == args.epochs) or \
                      (args.save_some_checkpoint and (epoch + 1) in args.save_epochs_lst)

        if should_save and args.output_dir:
            if args.use_deepspeed:
                model.save_checkpoint(args.output_dir, tag=f'checkpoint_{epoch}')
            else:
                payload = {'model': model_without_ddp.state_dict(), 'epoch': epoch}
                utils.save_on_master(payload, output_dir / f'checkpoint_{epoch}.pth')

        dev_stats = evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev', current_epoch=epoch) if dev_dataloader else None
        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, phase='test', current_epoch=epoch)

        metric_key = "top1_acc_pi"
        current_score = test_stats.get(metric_key, 0.0)
        
        if current_score > max_accuracy:
            max_accuracy = current_score
            rank0_print(f"*** New best {metric_key}: {current_score:.2f} (Epoch {epoch}) ***")
            if args.output_dir:
                if args.use_deepspeed:
                    model.save_checkpoint(args.output_dir, tag='best_checkpoint')
                else:
                    utils.save_on_master({'model': model_without_ddp.state_dict(), 'best_acc': max_accuracy}, 
                                   output_dir / 'best_checkpoint.pth')

        rank0_print(f'Current best (test) {metric_key.upper()}: {max_accuracy:.2f}')

        if utils.is_main_process():
            rank0_print(f'Current best (test) {metric_key.upper()}: {max_accuracy:.2f}')
            
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch}
            if dev_stats:
                log_stats.update({f'dev_{k}': v for k, v in dev_stats.items()})

            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.debug and epoch > 3:
            break

    total_time = time.time() - start_time
    rank0_print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))

    if args.eval:
        dev_stats = evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev') if dev_dataloader else None
        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, phase='test') if test_dataloader else None

    rank0_print('Done')

    if torch.distributed.is_initialized():
        time.sleep(5) 
        torch.distributed.barrier() 
        torch.distributed.destroy_process_group()    
    os._exit(0)

def perform_evaluation(args, dev_dataloader, test_dataloader, model, model_without_ddp, n_parameters, output_dir):
    if utils.is_main_process():
        partition = ''
        stats_lst = []
        if args.get_dev_result and dev_dataloader:
            rank0_print("DEV result")
            dev_stats = evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
            stats_lst.append([(k, v) for k, v in dev_stats.items()])
            partition += ' DEV '
        rank0_print("TEST result")
        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, phase='test')
        stats_lst.append([(k, v) for k, v in test_stats.items()])
        partition += ' TEST '

        log_stats = {'partition': partition, 'stats': str(stats_lst), 'n_parameters': n_parameters}
        
        if args.output_dir:
            with (output_dir / "eval_out.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    rank0_print("Eval done")

def recursive_to_float_cuda(data):
    if isinstance(data, torch.Tensor):
        return data.to(torch.float32).cuda()
    elif isinstance(data, dict):
        return {k: recursive_to_float_cuda(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_to_float_cuda(v) for v in data]
    else:
        return data

def train_one_epoch(args, model, model_without_ddp, data_loader, optimizer, epoch, hyp_optimizer=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    if optimizer is not None: optimizer.zero_grad()
    if hyp_optimizer: hyp_optimizer.zero_grad(set_to_none=True)

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        for key in src_input.keys():
            if isinstance(src_input[key], torch.Tensor):
                src_input[key] = recursive_to_float_cuda(src_input[key])

        stack_out = model(src_input, tgt_input)
        total_loss = stack_out['loss']

        if args.use_deepspeed:
            model.backward(total_loss)
        else:
            total_loss.backward()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            rank0_print(f"Warning: Loss contains {total_loss.item()}! Skipping this batch.")
            if optimizer is not None: optimizer.zero_grad()
            continue

        metric_logger.update(loss=total_loss.item())
        metric_logger.update(lm_loss=stack_out['lm_loss'].item())
        if "hyperbolic" in args.loop.lower():
            metric_logger.update(hy_loss=stack_out['hyp_loss'].item())

        if args.manifold == 'Lorentz' and hyp_optimizer and args.clip_grad_norm_hyp > 0:
            hyp_params_list = [p for p_group in hyp_optimizer.param_groups for p in p_group['params'] if p.requires_grad]
            valid_hyp_grads = [p for p in hyp_params_list if p.grad is not None]
            if valid_hyp_grads:
                clip_grad_norm_(valid_hyp_grads, max_norm=args.clip_grad_norm_hyp)

        if hyp_optimizer: 
            hyp_optimizer.step()
        
        if args.use_deepspeed:
            model.step()
        else:
            if optimizer is not None: optimizer.step()

        if hyp_optimizer and hasattr(model_without_ddp, 'global_step'):
            model_without_ddp.global_step += 1

        if hyp_optimizer:
            hyp_optimizer.zero_grad(set_to_none=True)
            metric_logger.update(hyp_lr=hyp_optimizer.param_groups[0]["lr"])

        metric_logger.update(loss=loss_value)
        if optimizer is not None:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        elif args.use_deepspeed:
            metric_logger.update(lr=model.get_lr()[0])
        
        if args.debug and step > 3: 
            break

    metric_logger.synchronize_between_processes()
    rank0_print("Averaged stats:", metric_logger)
    torch.cuda.empty_cache()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(args, data_loader, model, model_without_ddp, phase, current_epoch=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'{phase.upper()}:'

    local_pres = []
    local_refs = []
    
    world_size = utils.get_world_size()

    with torch.no_grad():
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            for key in src_input.keys():
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = recursive_to_float_cuda(src_input[key])

            stack_out = model(src_input, tgt_input)
            metric_logger.update(loss=stack_out['loss'].item())
            metric_logger.update(lm_loss=stack_out['lm_loss'].item())
            metric_logger.update(hyp_loss=stack_out['hyp_loss'].item())

            output = model_without_ddp.generate(
                stack_out, 
                max_new_tokens=100, 
                num_beams=4,
                synced_gpus=args.use_deepspeed
            )

            for o_tensor, ref_str in zip(output, tgt_input['gt_sentence']):
                local_pres.append(o_tensor.detach().cpu())
                local_refs.append(ref_str)
                
            if args.debug and step > 3:
                break

    if world_size > 1:
        torch.distributed.barrier()

        all_refs_list = [None for _ in range(world_size)]
        all_pres_list = [None for _ in range(world_size)]
        
        rank0_print(f"Gathering results from all {world_size} ranks...")
        torch.distributed.all_gather_object(all_refs_list, local_refs)
        torch.distributed.all_gather_object(all_pres_list, local_pres)
        
        global_refs = [item for sublist in all_refs_list for item in sublist]
        global_pres_tensors = [item for sublist in all_pres_list for item in sublist]
    else:
        global_refs = local_refs
        global_pres_tensors = local_pres

    tokenizer = model_without_ddp.mt5_tokenizer
    padding_value = tokenizer.eos_token_id

    top1_acc_pi, top1_acc_pc = 0.0, 0.0
    if len(global_pres_tensors) > 0:
        global_pres_padded = pad_sequence(global_pres_tensors, batch_first=True, padding_value=padding_value)
        global_pres_text = tokenizer.batch_decode(global_pres_padded, skip_special_tokens=True)
        
        total_len = len(data_loader.dataset)
        global_refs = global_refs[:total_len]
        global_pres_text = global_pres_text[:total_len]

        top1_acc_pi, top1_acc_pc = islr_performance(global_refs, global_pres_text)

    metric_logger.add_meter('top1_acc_pi', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('top1_acc_pc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    
    metric_logger.update(top1_acc_pi=top1_acc_pi)
    metric_logger.update(top1_acc_pc=top1_acc_pc)
    
    metric_logger.synchronize_between_processes()
    rank0_print(f"Averaged stats ({phase}):", metric_logger)

    if utils.is_main_process() and args.output_dir:
        out_path = Path(args.output_dir)
        log_file = out_path / f"{phase}_eval_log.txt"
        with open(log_file, "a") as f:
            info = current_epoch if current_epoch is not None else phase
            f.write(f"Epoch: {info} - PI: {top1_acc_pi:.4f}, PC: {top1_acc_pc:.4f}\n")

        if args.eval:
            with open(os.path.join(args.output_dir, f'{phase}_tmp_pres.txt'), 'w') as f:
                for r in global_pres_text: f.write(r + '\n')
            with open(os.path.join(args.output_dir, f'{phase}_tmp_refs.txt'), 'w') as f:
                for r in global_refs: f.write(r + '\n')

    torch.cuda.empty_cache()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Loop training scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()

    args.learnable_s = True
    args.learnable_c = True
    
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        args_log_path = output_path / "args.log"
        with open(args_log_path, 'w', encoding='utf-8') as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
                
    main(args)