"""
This file is modified from:
https://github.com/facebookresearch/deit/blob/main/utils.py
"""

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time, random
import numpy as np
from collections import defaultdict, deque
import datetime

import torch
from torch import Tensor
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

import pickle
import gzip

import argparse

# global definition
WORD_MASK = "<mask>"

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        # return self.total / self.count
        return self.total / self.count if self.count != 0 else 0
    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t", omit=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.omit = omit

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if str(name) == self.omit:
                continue
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def count_parameters_in_MB(model):
    # sum(p.numel() for p in model.parameters() if p.requires_grad)
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        print("save ckpt begin")
        torch.save(*args, **kwargs)
        print("save ckpt finish")

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    # print('args.distributed:', args.distributed)
    # print('args.rank:', args.rank)
    # print('args.world_size:', args.world_size)
    # print('args.gpu:', args.gpu)
    
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(f"Invalid GPU ID: {args.gpu}. Available GPUs: {torch.cuda.device_count()}")
    
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    local_rank = int(os.environ["LOCAL_RANK"])    

    torch.cuda.set_device(local_rank)    
    setup_for_distributed(args.rank == 0)
    print(f"Initialized rank {dist.get_rank()} on device {torch.cuda.current_device()}")


def sampler_func(clip, sn, random_choice=True):
    if random_choice:
        f = lambda n: [(lambda n, arr: n if arr == [] else np.random.choice(arr))(n * i / sn,
                                                                                range(int(n * i / sn),
                                                                                        max(int(n * i / sn) + 1,
                                                                                            int(n * (
                                                                                                    i + 1) / sn))))
                        for i in range(sn)]
    else:
        f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn, range(int(n * i / sn),
                                                                                                max(int(
                                                                                                    n * i / sn) + 1,
                                                                                                    int(n * (
                                                                                                            i + 1) / sn))))
                        for i in range(sn)]
    return f(clip)

def cosine_scheduler(base_value, final_value, epochs):
    iters = np.arange(epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    return schedule

def cosine_scheduler_func(base_value, final_value, iters, epochs):
    schedule = lambda x: final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * x / epochs))
    return schedule(iters)

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

def yield_tokens(file_path):
    with io.open(file_path, encoding = 'utf-8') as f:
        for line in f:
            yield line.strip().split()

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    
    output = torch.cat(tensors_gather,dim=0)
    return output

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    
    cudnn.deterministic = True # Since the input dim is dynamic.
    cudnn.benchmark = False # Since the input dim is dynamic.

def get_args_parser():
    parser = argparse.ArgumentParser('Uni-Sign scripts', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int)
    parser.add_argument('--gradient_clipping', default=1, type=float)
    parser.add_argument('--epochs', default=10, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=-100, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=-100, type=int)
    parser.add_argument('--gpu', default=-100, type=int)
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument("--hidden_dim", default=256, type=int)
    # * Finetuning params
    parser.add_argument('--f', type=str, help='Jupyter kernel JSON file', default=None)  
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--finetune_w_vlp', default=False, help='finetune from checkpoint with vlp', type=bool)

    # * Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1.0e-09)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: [0.9, 0.98], use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1.0e-3,
                        help='weight decay (default: 0.05)')
    
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1.0e-6, metavar='LR',
                        help='learning rate (default: 1e-6)')
    parser.add_argument('--sgt_dec_lr', type=float, default=1.0e-6, metavar='SGT_DEC_LR', 
                        help='learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1.0e-20, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=float, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

     # * Baise params
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save_checkpoint', type=bool, default=False)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dev_test_mode', type=bool, default=False)

    parser.add_argument('--num_workers', default=64, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--trick', type=bool, default=False)

    # deepspeed features
    parser.add_argument('--use_deepspeed', type=bool, default=False)
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                        type=str,
                        default='bf16',
                        choices=['fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument('--zero_stage',
                        type=int,
                        default=2,
                        help='ZeRO optimization stage for Actor model (and clones).')
    ## low precision
    parser.add_argument('--compute_fp32_loss',
                        action='store_true',
                        help='Relevant for low precision dtypes (fp16, bf16, etc.). '
                        'If specified, loss is calculated in fp32.')
    
    parser.add_argument('--quick_break',
                        type=int,
                        default=0,
                        help='save ckpt per quick_break step')
    
    # RGB branch
    parser.add_argument('--rgb_support', action='store_true',)
    
    # Pose length
    parser.add_argument("--max_length", default=256, type=int)
    
    # select dataset
    parser.add_argument("--dataset", default="CSL_Daily", choices=['CSL_News', "CSL_Daily", "WLASL"])
    parser.add_argument('--from_gpu', type=bool, default=False)
    
    # select task
    parser.add_argument("--task", default="SLT", choices=['SLT', "ISLR", "CSLR"])
    
    # select label smooth
    parser.add_argument("--label_smoothing", default=0.2, type=float)

    # select noise to mlm 
    parser.add_argument('--noise-rate', default=0.15, type=float)
    parser.add_argument('--noise-type', default='omit_last', type=str, choices=['omit', 'omit_last'])
    parser.add_argument('--random-shuffle', default=False, type=bool)

    # Cluster
    parser.add_argument('--use_cluster', default=True, type=bool)

    # vlp 
    parser.add_argument("--fusion_layer", default=3, type=int)
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='alpha for hcl_loss (default: 0.5)')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='beta for sg_loss (default: 0.5)')     
    parser.add_argument("--which_cross_attn", default=0, type=int)
    parser.add_argument("--strat_self_attn_ly", default=-1, type=int)
    parser.add_argument("--end_cross_attn_enc_ly", default=2, type=int)
    parser.add_argument('--zh_enitity', type=str, default='./entity/zh_entity_dict.txt')
    parser.add_argument('--en_enitity', type=str, default='./entity/en_entity_dict.txt')
    parser.add_argument('--loss_fct', type=str, default='NLLLoss', choices=['NLLLoss', 'KLLoss', 'CELoss'], help='For hcl_loss')
    parser.add_argument('--row_strategy', type=str, default='row_max', choices=['row_max', 'row_avg', 'row_topk_avg', 'row_softmax_weighted'],  help='For hcl_loss')
    parser.add_argument('--score_strategy', type=str, default='softmax', choices=['sum', 'average', 'logsumexp',  'softmax', 'var_reduced'],  help='For hcl_loss')
    parser.add_argument('--enc_hidden_state', default='vis', type=str)
    parser.add_argument('--freeze_txt_enc', default=False, type=bool)

    # ablation study
    return parser

class KLLoss(torch.nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    # def __init__(self, error_metric=torch.nn.KLDivLoss(size_average=True, reduce=True)):
    def __init__(self, error_metric=torch.nn.KLDivLoss(reduction='mean')):
        super().__init__()
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss
        
def loss_fn_kd(outputs, teacher_outputs, T=1.0, alpha=0.5):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss = torch.nn.KLDivLoss( reduction='sum')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (T * T) #+ \
            #    F.cross_entropy(outputs, F.softmax(teacher_outputs, dim=1)) * (1. - alpha)

    return KD_loss

def noise_injecting(raw_gloss, noise_rate=0.15, noise_type='omit_last', random_shuffle=False, is_train=True):
    new_gloss = []

    for ii, gloss in enumerate(raw_gloss):
        text = gloss.split()

        if noise_type == 'omit':
            # del noise
            if random.uniform(0, 1) <= 1. and is_train:
                index = sampler_func(len(text), int(len(text)*(1. - noise_rate)), random_choice=is_train)
                noise_gloss = []
                noise_idx = []
                for i, d in enumerate(text):
                    if i in index:
                        noise_gloss.append(d)
                    else:
                        noise_gloss.append(WORD_MASK)
                        noise_idx.append(i)
            else:
                noise_gloss = [d for d in text]

        elif noise_type == 'omit_last' :
            if random.uniform(0, 1) <= 1.0 and is_train:
                index = np.arange(0, len(text) - int(np.ceil(len(text)*(np.random.uniform(0,noise_rate,(1,))))), 1, dtype=int)
                noise_gloss = []
                for i, d in enumerate(text):
                    if i in index:
                        noise_gloss.append(d)
                    else:
                        noise_gloss.append(WORD_MASK)
            else:
                noise_gloss = [d for d in text]
        
        if is_train and random_shuffle and random.uniform(0, 1) > 0.5:
            random.shuffle(noise_gloss) # random shuffle sequence

        new_gloss.append(' '.join(noise_gloss))
    return new_gloss

