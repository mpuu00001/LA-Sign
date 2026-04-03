import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
import os
import random
import numpy as np
import copy
import pickle
import re
from config import pose_dirs 

def load_part_kp(skeletons, confs, force_ok=False):
    thr = 0.3
    kps_with_scores = {}
    scale = None
    
    for part in ['body', 'left', 'right', 'face_all']:
        kps = []
        confidences = []
        
        for skeleton, conf in zip(skeletons, confs):
            skeleton = skeleton[0]
            conf = conf[0]
            
            if part == 'body':
                hand_kp2d = skeleton[[0] + [i for i in range(3, 11)], :]
                confidence = conf[[0] + [i for i in range(3, 11)]]
            elif part == 'left':
                hand_kp2d = skeleton[91:112, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[91:112]
            elif part == 'right':
                hand_kp2d = skeleton[112:133, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[112:133]
            elif part == 'face_all':
                hand_kp2d = skeleton[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53], :]
                hand_kp2d = hand_kp2d - hand_kp2d[-1, :]
                confidence = conf[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53]]
            else:
                raise NotImplementedError
            
            kps.append(hand_kp2d)
            confidences.append(confidence)
            
        kps = np.stack(kps, axis=0)
        confidences = np.stack(confidences, axis=0)
        
        if part == 'body':
            if force_ok:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[...,None]], axis=-1), thr)
            else:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[...,None]], axis=-1), thr)
        else:
            assert not scale is None
            result = np.concatenate([kps, confidences[...,None]], axis=-1)
            if scale==0:
                result = np.zeros(result.shape)
            else:
                result[...,:2] = (result[..., :2]) / scale
                result = np.clip(result, -1, 1)
                result[result[...,2]<=thr] = 0
            
        kps_with_scores[part] = torch.tensor(result)
        
    return kps_with_scores

def crop_scale(motion, thr):
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]>thr][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    ratio = 1
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2] - [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    result[result[...,2]<=thr] = 0
    return result, scale, [xs,ys]

class Base_Dataset(Dataset.Dataset):
    def collate_fn(self, batch):
        tgt_batch,src_length_batch,name_batch,pose_tmp,gloss_batch,label_key_batch = [],[],[],[],[],[]
        
        # Note: The 5th element used to be support_rgb_dict, now it receives None
        for name_sample, pose_sample, text, gloss, _, label_key, _ in batch:
            name_batch.append(name_sample)
            pose_tmp.append(pose_sample)
            tgt_batch.append(text)
            gloss_batch.append(gloss)
            label_key_batch.append(label_key)
            
        src_input = {}
        keys = pose_tmp[0].keys()

        for key in keys:
            max_len = max([len(vid[key]) for vid in pose_tmp])
            video_length = torch.LongTensor([len(vid[key]) for vid in pose_tmp])
            
            padded_video = [torch.cat(
                (
                    vid[key],
                    vid[key][-1][None].expand(max_len - len(vid[key]), -1, -1),
                )
                , dim=0)
                for vid in pose_tmp]
            
            img_batch = torch.stack(padded_video,0)
            
            src_input[key] = img_batch
            if 'attention_mask' not in src_input.keys():
                src_length_batch = video_length

                mask_gen = []
                for i in src_length_batch:
                    tmp = torch.ones([i]) + 7
                    mask_gen.append(tmp)
                mask_gen = pad_sequence(mask_gen, padding_value=0,batch_first=True)
                img_padding_mask = (mask_gen != 0).long()
                src_input['attention_mask'] = img_padding_mask

                src_input['name_batch'] = name_batch
                src_input['src_length_batch'] = src_length_batch
                
        tgt_input = {}
        tgt_input['gt_sentence'] = tgt_batch
        tgt_input['gt_gloss'] = gloss_batch
        tgt_input['label_key'] = label_key_batch

        return src_input, tgt_input

class S2T_Dataset(Base_Dataset):
    def __init__(self, path, args, phase):
        super(S2T_Dataset, self).__init__()
        self.args = args
        self.max_length = args.max_length
        self.raw_data = utils.load_dataset_file(path)
        self.phase = phase

        if "WLASL" in self.args.dataset or "MSASL" in self.args.dataset:
            dataset_name = re.sub(r'\d+', '', args.dataset)
            self.pose_dir = os.path.join(pose_dirs[dataset_name], phase)
        else:
            raise NotImplementedError("Only WLASL and MSASL datasets are supported.")

        self.list = list(self.raw_data.keys())

        if getattr(self.args, "test", False):
            print(f"[INFO] Test mode active — limiting dataset to first 38 samples (out of {len(self.list)})")
            self.list = self.list[:38]
        
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]

        text = sample['text']
        if "gloss" in sample.keys():
            gloss = " ".join(sample['gloss'])
        else:
            gloss = ''
        
        name_sample = sample['name']
        
        # WLASL/MSASL
        pose_sample, _ = self.load_pose(sample['video_path'])

        cated_pose_sample = [pose_sample[p] for p in ['left','body','face_all','right']]         
        cated_pose_sample = [t.contiguous() for t in cated_pose_sample]       
        cated_pose_sample = torch.cat(cated_pose_sample, dim=1) 

        return name_sample, pose_sample, text, gloss, None, key, None
    
    def load_pose(self, path):
        if 'MSASL' in self.args.dataset:
            path = path.split("/")[-1]

        if 'WLASL300' in self.args.dataset:   
            path = path.split("/")[-1].split('-')[0] + '.pkl'
    
        pose = pickle.load(open(os.path.join(self.pose_dir, path.replace(".mp4", '.pkl')), 'rb'))
            
        if 'start' in pose.keys():
            assert pose['start'] < pose['end']
            duration = pose['end'] - pose['start']
            start = pose['start']
        else:
            duration = len(pose['scores'])
            start = 0
                
        if duration > self.max_length:
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))
        
        tmp = np.array(tmp) + start
            
        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp
    
        kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)

        return kps_with_scores, None

    def __str__(self):
        return f'# {self.phase} with a total of {len(self)}'