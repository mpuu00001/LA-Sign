import torch
import torch.nn as nn
from transformers import MT5ForConditionalGeneration 
from typing import Optional
from config import mt5_path, mt5_aux_path
import torch.nn.functional as F
from utils import KLLoss

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
        sim_matrix = Q @ D.permute(0, 2, 1)  
        sim_values = compute_similarity_values(sim_matrix, strategy=row_strategy)
        sim_score = compute_similarity_score(sim_values, strategy=score_strategy)
        return sim_score

    assert similarity_metric == 'l2'
    return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1)) ** 2).sum(-1)).max(-1).values.sum(-1)

def sign2text_sim_martix(text_feats, vis_feats, args):
    num_text, num_sign = text_feats.shape[0], vis_feats.shape[0]
    sim_s2t = torch.zeros((num_sign, num_text)).to(text_feats.device)
    for i in range(num_sign):
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
    sim_s2t = logit_scale * sign_feature @ text_feature.t()
    sim_t2s = logit_scale * text_feature @ sign_feature.t()
    
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
