import torch
import torch.nn as nn
from transformers import MT5ForConditionalGeneration 
from typing import Optional
from config import mt5_path
import torch.nn.functional as F
from utils import KLLoss
import numpy as np
from sklearn.decomposition import PCA
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def is_torchdynamo_compiling():
    # Importing torch._dynamo causes issues with PyTorch profiler (https://github.com/pytorch/pytorch/issues/130622)
    # hence rather relying on `torch.compiler.is_compiling()` when possible (torch>=2.3)
    try:
        import torch

        return torch.compiler.is_compiling()
    except Exception:
        try:
            import torch._dynamo as dynamo  # noqa: F401

            return dynamo.is_compiling()
        except Exception:
            return False

class X_CrossAttn(nn.Module):

    def __init__(self, args):
        super(X_CrossAttn, self).__init__()

        decdr_ly = MT5ForConditionalGeneration.from_pretrained(mt5_path).get_decoder().block[args.which_cross_attn].layer[1]
        self.cross_att = decdr_ly.EncDecAttention
        
    def forward(self, vis_hidden_states, txt_hidden_states): # vis, txt

        batch_size, vis_length = vis_hidden_states.shape[:2]
        _, tgt_length = txt_hidden_states.shape[:2]

        query_states = self.cross_att.q(vis_hidden_states)
        query_states = query_states.view(batch_size, -1, self.cross_att.n_heads, self.cross_att.key_value_proj_dim).transpose(1, 2) # Vis_q

        key_states = self.cross_att.k(txt_hidden_states)
        key_states = key_states.view(batch_size, -1, self.cross_att.n_heads, self.cross_att.key_value_proj_dim).transpose(1, 2) # Txt_q

        value_states = self.cross_att.v(txt_hidden_states)
        value_states = value_states.view(batch_size, -1, self.cross_att.n_heads, self.cross_att.key_value_proj_dim).transpose(1, 2) # Txt_v

        value_states2 = self.cross_att.v(vis_hidden_states)
        value_states2 = value_states2.view(batch_size, -1, self.cross_att.n_heads, self.cross_att.key_value_proj_dim).transpose(1, 2) # Vis_v

        scores = torch.matmul(query_states, key_states.transpose(3, 2)) # Vis_q * (Txt_q)T

        key_length = key_states.shape[-2]

        position_bias = torch.zeros(
                (1, self.cross_att.n_heads, vis_length, key_length), device=scores.device, dtype=scores.dtype
            )
        if self.cross_att.gradient_checkpointing and self.training:
            position_bias.requires_grad = True

        scores = scores + position_bias
        
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)   # attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        vis_attn_output = torch.matmul(attn_weights, value_states) # Attn * Txt_v

        attn_weights_t = attn_weights.transpose(2, 3)    
        txt_attn_output = torch.matmul(attn_weights_t, value_states2) # Attn^T * Vis_v 

        vis_attn_output = vis_attn_output.transpose(1, 2).contiguous()
        vis_attn_output = vis_attn_output.view(batch_size, -1, self.cross_att.inner_dim)

        txt_attn_output = txt_attn_output.transpose(1, 2).contiguous()
        txt_attn_output = txt_attn_output.view(batch_size, -1, self.cross_att.inner_dim)

        vis_attn_output = self.cross_att.o(vis_attn_output)
        txt_attn_output = self.cross_att.o(txt_attn_output)

        return vis_attn_output, txt_attn_output


class EmbeddingClusterHelperAutomaton:
    def __init__(self, tokenizer, dict_path, masked_token=None):
        self.tokenizer = tokenizer
        self.entity_ids_dict = self.load_dict(dict_path)
        self.masked_ids = [[i] for i in tokenizer.convert_tokens_to_ids(masked_token)]

    def load_dict(self, dict_path):
        import ahocorasick

        entity_ids_dict = ahocorasick.Automaton()

        # with open("zh_entity_dict.txt", "w", encoding="utf8") as f:
        for i, line in enumerate(open(dict_path, encoding="utf8")):
            entity_ids = self.tokenizer.encode(line.strip(), add_special_tokens=False)
            entity_ids = entity_ids[1:]
            if len(entity_ids) > 1:
                entity_ids_dict.add_word(str(tuple(entity_ids)), line.strip())
                    # f.write(line.strip() + "\n")  # Save to file

        entity_ids_dict.make_automaton()
        return entity_ids_dict

    def ahocorasick_maximum_matching(self, input_ids):
        offsets = [[]]
        words = [[]]
        input_ids_tmp = [x.item() for x in input_ids]

        for i, idx in enumerate(input_ids_tmp):                
            key = str(tuple(words[-1] + [idx]))
            # print("key in matching", key)
            if self.entity_ids_dict.match(key):
                words[-1] += [idx]
                offsets[-1] += [i+1]
            else:
                words.append([idx])
                offsets.append([i])
        return [(o[0], o[-1]) for o in offsets if len(o) > 1]

    def pad_offsets(self, offsets_sort, input_len):
        # step 0: if offset list is empty, return range
        if not offsets_sort:
            return list(range(input_len))
        # step 1: pad index before first offset
        first_offset = offsets_sort[0]
        offsets_pad = [i for i in range(first_offset[0])]
        # step 2: pad index between
        group_size = len(offsets_sort)
        for i in range(len(offsets_sort)):
            offsets_pad.append(offsets_sort[i][0])
            offsets_pad.append(offsets_sort[i][1])
            if i + 1 < group_size:
                first_end = offsets_sort[i][1]
                second_start = offsets_sort[i + 1][0]
                for j in range(first_end + 1, second_start):
                    offsets_pad.append(j)
        # step 3: pad index last
        last_offset = offsets_sort[-1]
        for k in range(last_offset[1] + 1, input_len):
            offsets_pad.append(k)
        return offsets_pad

    def get_offsets(self, input_ids):
        offsets = self.ahocorasick_maximum_matching(input_ids)
        return self.pad_offsets(offsets, len(input_ids))

    def get_embed_cluster_input_ids(self, input_ids, embed_cluster_offset):
        embed_cluster_input_ids = []
        for i, start in enumerate(embed_cluster_offset):
            if i + 1 < len(embed_cluster_offset):
                end = embed_cluster_offset[i + 1]
                embed_cluster_input_ids.append(input_ids[start:end])
            else:
                embed_cluster_input_ids.append(input_ids[start:])
        return embed_cluster_input_ids

    def get_embed_cluster_attn_mask(self, one_input_ids):
        one_attn_mask = []
        for input_id in one_input_ids:
            if input_id not in self.masked_ids:
                one_attn_mask.append(1)
            else:
                one_attn_mask.append(0)
        return one_attn_mask

    def process(self, text_input, return_mask=False):
        input_offsets = [self.get_offsets(i) for i in text_input.input_ids]
        min_len = min([len(offset) for offset in input_offsets])
        # truncation
        input_offsets_truncated = [offset[:min_len] for offset in input_offsets]
        # flatten to 1d
        input_offsets_flatten = []
        text_len = len(text_input.input_ids[0])
        for i, offset in enumerate(input_offsets_truncated):
            input_offsets_flatten.extend([o + i * text_len for o in offset])
        # return
        if return_mask:
            embed_cluster_input_ids = [
                self.get_embed_cluster_input_ids(i, o)
                for i, o in zip(text_input.input_ids, input_offsets)
            ]
            attn_mask_list = [
                self.get_embed_cluster_attn_mask(i) for i in embed_cluster_input_ids
            ]
            attn_mask_truncated = [mask[:min_len] for mask in attn_mask_list]
            return input_offsets_flatten, attn_mask_truncated
        return input_offsets_flatten

def compute_similarity_score(sim_values, strategy='sum'):
    if strategy == 'sum':
        return sim_values.sum(dim=1)
    elif strategy == 'average':
        return sim_values.mean(dim=1)
    elif strategy == 'softmax':
        weights = F.softmax(sim_values, dim=1)  # [B, N]
        return (sim_values * weights).sum(dim=1)
    elif strategy == 'logsumexp':
        return torch.logsumexp(sim_values, dim=1)  # More numerically stable than exp().sum().log()

    elif strategy == 'var_reduced':
        mean = sim_values.mean(dim=1, keepdim=True)
        centered = sim_values - mean
        return centered.sum(dim=1)
    else:
        raise NotImplementedError

def compute_similarity_values(sim_matrix, strategy='row_max'):
    if strategy == 'row_max':
        return sim_matrix.max(dim=2).values  
    elif strategy == 'row_avg':
       return sim_matrix.mean(dim=2)  
    elif strategy == 'row_topk_avg':
        k = max(1, int(sim_matrix.shape[2] / 3))  # ensure k ≥ 1
        topk_vals, _ = torch.topk(sim_matrix, k=k, dim=2)
        return topk_vals.mean(dim=2)  
    elif strategy == 'row_softmax_weighted':
        weights = F.softmax(sim_matrix, dim=2)
        return (sim_matrix * weights).sum(dim=2)  
    else:
        raise NotImplementedError

def tokenwise_similarity(Q, D, row_strategy='row_max', score_strategy='sum', similarity_metric='cosine'):
    if similarity_metric == 'cosine':
        sim_matrix = Q @ D.permute(0, 2, 1)  # [B, S/T, T/S]
        sim_values = compute_similarity_values(sim_matrix, strategy=row_strategy) # [B S/T]
        sim_score = compute_similarity_score(sim_values, strategy=score_strategy) # [B]
        return sim_score

    assert similarity_metric == 'l2'
    return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1)) ** 2).sum(-1)).max(-1).values.sum(-1)

def sign2text_sim_martix(text_feats, vis_feats, args):
    num_text, num_sign = text_feats.shape[0], vis_feats.shape[0]
    sim_s2t = torch.zeros((num_sign, num_text)).to(text_feats.device)
    for i in range(num_sign): # positive pairs vis_feats[i] and text_feats[i], negative pairs vis_feats[i] and text_feats[j]
        row_sim = tokenwise_similarity(vis_feats[i], text_feats, 
                                       row_strategy=args.row_strategy, 
                                       score_strategy=args.score_strategy)
        sim_s2t[i] = row_sim
    return sim_s2t

def text2sign_sim_martix(text_feats, vis_feats, args):
    num_text, num_sign = text_feats.shape[0], vis_feats.shape[0]
    sim_t2s = torch.zeros((num_text, num_sign)).to(text_feats.device)
    for i in range(num_text):
        row_sim = tokenwise_similarity(text_feats[i], vis_feats,
                                       row_strategy=args.row_strategy, 
                                       score_strategy=args.score_strategy)
        sim_t2s[i] = row_sim
    return sim_t2s

def tokenwise_similarity_martix(text_feats, vis_feats, args):
    sim_s2t = sign2text_sim_martix(text_feats, vis_feats, args)
    sim_t2s = text2sign_sim_martix(text_feats, vis_feats, args)
    return sim_s2t, sim_t2s

def NLLLoss_SN(sim, targets):
    loss = -torch.sum(F.log_softmax(sim, dim=1) * targets, dim=1).mean()
    return loss

def Kl(sim, target):
    loss_fct = KLLoss()
    loss = loss_fct(sim, target)
    return loss

def CE(sim, target):
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(sim, target)
    return loss

def tokenwise_similarity_loss(vis_feats, text_feats, args):
    sim_s2t, sim_t2s = tokenwise_similarity_martix(text_feats, vis_feats, args)

    s2t_targets = torch.zeros_like(sim_s2t).to(sim_t2s.device)
    s2t_targets.fill_diagonal_(1)

    t2s_targets = torch.zeros_like(sim_t2s).to(sim_t2s.device)
    t2s_targets.fill_diagonal_(1)

    if args.loss_fct == 'NLLLoss':
        loss_s2t = -torch.sum(F.log_softmax(sim_s2t, dim=1) * s2t_targets, dim=1).mean()
        loss_t2s = -torch.sum(F.log_softmax(sim_t2s, dim=1) * t2s_targets, dim=1).mean()
    elif args.loss_fct == 'KLLoss':
        loss_s2t = Kl(sim_s2t, s2t_targets)
        loss_t2s = Kl(sim_t2s, t2s_targets)
    elif args.loss_fct == 'CELoss':
        loss_s2t = CE(sim_s2t, s2t_targets)
        loss_t2s = CE(sim_t2s, t2s_targets)
    else:
        raise NotImplementedError
    
    return (loss_s2t + loss_t2s) / 2

def gloabal_similarity_loss(vis_global_token, text_global_token, logit_scale, args):
    sign_feature = vis_global_token
    text_feature = text_global_token

    logit_scale = logit_scale.exp()
    sim_s2t = logit_scale * sign_feature @ text_feature.t() # positive pairs matched sign-text pairs  # [B, B]
    sim_t2s = logit_scale * text_feature @ sign_feature.t() # negative pairs are unmatched/other sign-text pairs # [B, B]

    sim_targets = torch.zeros(sim_s2t.size()).to(sim_t2s.device)
    sim_targets.fill_diagonal_(1)

    if args.loss_fct == 'NLLLoss':
        loss_s2t = -torch.sum(F.log_softmax(sim_s2t, dim=1) * sim_targets, dim=1).mean()
        loss_t2s = -torch.sum(F.log_softmax(sim_t2s, dim=1) * sim_targets, dim=1).mean()
    elif args.loss_fct == 'KLLoss':
        loss_s2t = Kl(sim_s2t, sim_targets)
        loss_t2s = Kl(sim_t2s, sim_targets) 
    elif args.loss_fct == 'CELoss':
        loss_s2t = CE(sim_s2t, sim_targets)
        loss_t2s = CE(sim_t2s, sim_targets)
    else:    
        raise NotImplementedError
    
    return (loss_s2t + loss_t2s) / 2

class StepWarmUpScheduler(object):
    def __init__(self, start_ratio, end_ratio, warmup_start_step, warmup_step):
        super().__init__()
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.warmup_start_step = warmup_start_step
        self.warmup_step = warmup_step + int(warmup_step == 0)
        self.step_ratio = (end_ratio - start_ratio) / self.warmup_step

    def forward(self, step_num):
        if step_num < self.warmup_start_step:
            return self.start_ratio
        elif step_num >= self.warmup_step:
            return self.end_ratio
        else:
            ratio = self.start_ratio + self.step_ratio * (step_num - self.warmup_start_step)
            return ratio

class BiLSTMLayer(nn.Module):
    def __init__(self, input_size=768, debug=False, hidden_size=256, num_layers=2, dropout=0.3,
                 bidirectional=True, rnn_type='LSTM'):
        super(BiLSTMLayer, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = int(hidden_size / self.num_directions)
        self.rnn_type = rnn_type
        self.debug = debug
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True
        )

    def forward(self, src_feats, src_lens, max_len, hidden=None):
        """
        Args:
            - src_feats: (batch_size, max_src_len, 512)
            - src_lens: (batch_size)
        Returns:
            - outputs: (max_src_len, batch_size, hidden_size * num_directions)
            - hidden : (num_layers, batch_size, hidden_size * num_directions)
        """
        packed_emb = nn.utils.rnn.pack_padded_sequence(src_feats, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_emb)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=max_len)

        return rnn_outputs

class AdaptiveMask(nn.Module):
    """
    DDEM v3
    """

    def __init__(self, input_size=768, output_size=768, dropout=0.1):
        """
        AF module
        :param input_size: dimensionality of the input.
        :param output_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(AdaptiveMask, self).__init__()
        self.lstm = BiLSTMLayer(input_size=input_size, hidden_size=output_size, dropout=dropout)
        self.linear = nn.Linear(output_size, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(output_size, eps=1e-5)

    def forward(self, input_tensor, input_len, k=2, mask=None):
        lstm_o = self.lstm(input_tensor, input_len, input_tensor.shape[1])
        list_out = self.softmax(self.linear(lstm_o).squeeze(-1))
        _, indices = list_out.topk(k, dim=-1, largest=False, sorted=False)
        lstm_o = self.layer_norm(lstm_o)

        # update mask
        if mask is not None:
            sgn_mask_copy = mask.clone().detach()
            for b in range(input_tensor.shape[0]):
                sgn_mask_copy[b, indices[b]] = False
            return lstm_o, sgn_mask_copy
        else:
            return lstm_o

def sample_poisson_lognormal(target_mean, this_sigma=0.5):

    mu = torch.log(torch.tensor(target_mean)) - 0.5 * this_sigma**2
    
    lognormal_dist = torch.distributions.LogNormal(mu, this_sigma)
    lambda_sample = lognormal_dist.sample()

    poisson_dist = torch.distributions.Poisson(lambda_sample)
    return poisson_dist.sample().item()

class AdaptiveFusion(nn.Module):
    def __init__(self, input_size_1=768, input_size_2=768, output_siz=2, bias=False):
        super(AdaptiveFusion, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.weight_input_1 = nn.Linear(input_size_1, output_siz, bias=bias)
        self.weight_input_2 = nn.Linear(input_size_2, output_siz, bias=bias)
        self.layer_norm = nn.LayerNorm(input_size_1, eps=1e-5)

    def forward(self, input_1, input_2):
        fm_sigmoid = self.sigmoid(self.weight_input_1(input_1) + self.weight_input_2(input_2))
        lambda1 = fm_sigmoid.clone().detach()[:, :, 0].unsqueeze(-1)
        lambda2 = fm_sigmoid.clone().detach()[:, :, 1].unsqueeze(-1)

        fused_output = input_1 + input_2 + torch.mul(lambda1, input_1) + torch.mul(lambda2, input_2)
        fused_output = self.layer_norm(fused_output)
        return fused_output

class BiLSTMLayer(nn.Module):
    def __init__(self, input_size=768, hidden_size=768, num_layers=2, dropout=0.3,
                 bidirectional=True, rnn_type='LSTM'):
        super(BiLSTMLayer, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = int(hidden_size / self.num_directions)
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-5)

    def forward(self, src_feats, src_lens, max_len):
        packed_emb = nn.utils.rnn.pack_padded_sequence(src_feats, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.rnn(packed_emb)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=max_len)
        rnn_outputs = self.layer_norm(rnn_outputs)
        return rnn_outputs

def save_pca_projected_embeddings(args,hyp_body, hyp_left, hyp_right, hyp_face, save_path="pca_projected_embeddings.npz"):
    """
    Applies PCA to Poincaré embeddings and saves the 2D projected data.
    """
    def pca_project(tensor):
        x = tensor.detach().cpu().float().numpy()
        pca = PCA(n_components=2)
        return pca.fit_transform(x)

    body_proj  = pca_project(hyp_body)
    left_proj  = pca_project(hyp_left)
    right_proj = pca_project(hyp_right)
    face_proj  = pca_project(hyp_face)

    labels = (["body"] * len(body_proj) +
              ["left"] * len(left_proj) +
              ["right"] * len(right_proj) +
              ["face"] * len(face_proj))

    all_data = np.concatenate([body_proj, left_proj, right_proj, face_proj], axis=0)
    labels   = np.array(labels)
    save_path = f"/home/mupu0001/LoopSLU2025/vis/{args.dataset.lower()}/pca_projected_embeddings_{args.dataset}_{args.count}.npz"
    # Save as .npz for later loading
    np.savez(save_path, x=all_data[:, 0], y=all_data[:, 1], labels=labels)

    args.count += 1
    # Optionally: print the shape for debug
    print(f"Saved PCA projected data to {save_path} with shape {all_data.shape}")

def save_poincare_logmap_pca_S(args, hyp_body, hyp_left, hyp_right, hyp_face, manifold, save_path="pca_logmap_projected_embeddings.npz"):

    def logmap_and_project(tensor):
        logmap = manifold.logmap0(tensor.float())  # -> Euclidean
        x_np = logmap.detach().cpu().numpy()
        return PCA(n_components=2).fit_transform(x_np)

    body_proj  = logmap_and_project(hyp_body)
    left_proj  = logmap_and_project(hyp_left)
    right_proj = logmap_and_project(hyp_right)
    face_proj  = logmap_and_project(hyp_face)

    labels = (["body"] * len(body_proj) +
              ["left"] * len(left_proj) +
              ["right"] * len(right_proj) +
              ["face"] * len(face_proj))

    all_data = np.concatenate([body_proj, left_proj, right_proj, face_proj], axis=0)
    labels   = np.array(labels)

    # Optional: Normalize to unit disk
    radii = np.sqrt(np.sum(all_data**2, axis=1))
    max_radius = np.max(radii)
    if max_radius >= 1.0:
        print(f"[Warning] Max radius before normalization: {max_radius:.4f}, normalizing to fit inside unit disk.")
        all_data = all_data / (max_radius + 1e-6)

    # Print diagnostics
    # print(f"Projected radius range: min={radii.min():.4f}, max={radii.max():.4f}, mean={radii.mean():.4f}")

    # Construct path
    save_path = f"/home/mupu0001/LoopSLU2025/vis/{args.dataset.lower()}/pca_projected_embeddings_{args.dataset}_{args.count}.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save
    np.savez(save_path, x=all_data[:, 0], y=all_data[:, 1], labels=labels)
    # print(f"[✓] Saved PCA projected data to {save_path} with shape {all_data.shape}")

    args.count += 1


def save_poincare_logmap_pca_SandT(args, pose_mu_mfd, hyp_text, manifold, save_path="pca_logmap_projected_embeddings.npz"):

    def logmap_and_project(tensor):
        logmap = manifold.logmap0(tensor.float())  # -> Euclidean
        x_np = logmap.detach().cpu().numpy()
        return PCA(n_components=2).fit_transform(x_np)

    pose_proj  = logmap_and_project(pose_mu_mfd)
    text_proj  = logmap_and_project(hyp_text)

    labels = (["pose"] * len(pose_proj) +
              ["text"] * len(text_proj))

    all_data = np.concatenate([pose_proj, text_proj], axis=0)
    labels   = np.array(labels)

    # Optional: Normalize to unit disk
    radii = np.sqrt(np.sum(all_data**2, axis=1))
    max_radius = np.max(radii)
    if max_radius >= 1.0:
        print(f"[Warning] Max radius before normalization: {max_radius:.4f}, normalizing to fit inside unit disk.")
        all_data = all_data / (max_radius + 1e-6)

    # Print diagnostics
    print(f"Projected radius range: min={radii.min():.4f}, max={radii.max():.4f}, mean={radii.mean():.4f}")

    # Construct path
    save_path = f"/home/mupu0001/LoopSLU2025/vis/{args.dataset.lower()}/pose_text_feats/pca_projected_embeddings_{args.dataset}_{args.count}.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save
    np.savez(save_path, x=all_data[:, 0], y=all_data[:, 1], labels=labels)
    print(f"[✓] Saved PCA projected data to {save_path} with shape {all_data.shape}")

    args.count += 1

def is_main_process():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0


# ============================================================================ #
# Adapted from the hyperbolic contrastive regularisation of Geo-Sign
# ============================================================================ #

class HyperbolicMapper(nn.Module):
    def __init__(self, in_dim, out_dim, hyp):
        super().__init__()
        self.linear      = nn.Linear(in_dim, out_dim, bias=True)
        self.scale_param = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.hyp = hyp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cast to weight dtype for mixed precision compatibility
        x_cast   = x.to(self.linear.weight.dtype)
        # scale tangent vector before lifting onto manifold
        tangent  = self.linear(x_cast) * self.scale_param.exp()
        # lift to hyp all via exponential map at origin
        return self.hyp.expmap0(tangent.float(), project=True)


class HyperbolicAlignmentLoss(nn.Module):
    def __init__(self, label_smoothing, hyp):
        super().__init__()
        self.hyp = hyp 
        self.log_tau   = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.margin    = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=-100,
        )

    def _empty_output(self, device: torch.device) -> Dict:
        return {
            "loss":     torch.tensor(0.0, device=device, requires_grad=True),
            "pos_dist": torch.tensor(0.0, device=device),
            "tau":      torch.tensor(0.0, device=device),
            "margin":   self.margin.detach(),
        }

    def forward(self, sign_emb: torch.Tensor,
                text_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            sign_emb: (B, d) hyperbolic sign embeddings
            text_emb: (B, d) hyperbolic text embeddings
        """
        B = sign_emb.shape[0]
        if B == 0:
            return self._empty_output(sign_emb.device)

        # pairwise geodesic distances in hyperbolic space: (B, B)
        geo_dist = self.hyp.dist(
            sign_emb.unsqueeze(1).expand(-1, B, -1),
            text_emb.unsqueeze(0).expand(B, -1, -1),
            keepdim=False,
        )

        # similarity = negative geodesic distance scaled by temperature
        tau     = torch.sigmoid(self.log_tau) * 1.99 + 0.01
        scores  = -geo_dist / tau

        # apply margin to all off-diagonal (negative) pairs
        is_negative = ~torch.eye(B, device=scores.device, dtype=torch.bool)
        scores      = scores + self.margin.clamp(min=0.0) * is_negative.float()

        # contrastive loss with diagonal as positive pairs
        labels = torch.arange(B, device=sign_emb.device)
        loss   = self.criterion(scores, labels)

        return {
            "loss":     loss,
            "pos_dist": geo_dist.diagonal().mean().detach(),
            "tau":      tau.detach(),
            "margin":   self.margin.detach(),
        }


def compute_frechet_mean(
    part_embeddings: torch.Tensor,   # (N, B, d) hyperbolic part embeddings
    part_weights:    torch.Tensor,   # (N, B)    positive attention weights
    hyp,
    max_iter:        int   = 50,
    tol:             float = 1e-5, 
) -> torch.Tensor:                   # (B, d)
    """
    Args:
        part_embeddings: hyperbolic embeddings for each body part
        part_weights:    normalised weights per part per sample
        max_iter:        maximum Riemannian gradient steps
        tol:             convergence threshold on geodesic displacement
    """
    # normalise weights across parts dimension
    w   = part_weights / (part_weights.sum(dim=0, keepdim=True) + 1e-8)  # (N, B)
    # initialise mean at first part embedding
    mu  = part_embeddings[0].clone()   # (B, d)

    for _ in range(max_iter):
        # map all part embeddings to tangent space at current mean
        log_vecs  = hyp.logmap(
            mu.unsqueeze(0),      # (1, B, d)
            part_embeddings,      # (N, B, d)
        )                         # (N, B, d)

        # weighted sum of tangent vectors -> gradient direction
        grad_step = (w.unsqueeze(-1) * log_vecs).sum(dim=0)   # (B, d)

        # retract onto manifold via exponential map
        mu_next   = hyp.expmap(mu, grad_step, project=True)

        # check convergence via geodesic displacement
        delta     = hyp.dist(mu_next, mu, keepdim=False)
        if (delta < tol).all():
            break
        mu = mu_next

    return mu