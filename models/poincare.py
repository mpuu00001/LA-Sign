from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MT5ForConditionalGeneration, T5Tokenizer

import contextlib, math, warnings
from typing import Dict

import geoopt
from geoopt import ManifoldParameter
from geoopt.manifolds import PoincareBall, Euclidean, Lorentz, Scaled

from stgcn_layers import Graph, get_stgcn_chain
from config import mt5_path

from models.models import trunc_normal_
from models.tools import is_main_process
from models.tools import HyperbolicMapper, HyperbolicAlignmentLoss, compute_frechet_mean

class Uni_Sign_Hyperbolic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args    = args
        self.modes = ["body", "left", "right", "face_all"]
        initial_gcn_dim = 64

        self.graph, As, self.proj_linear = {}, [], nn.ModuleDict()
        for m in self.modes:
            g = Graph(layout=m, strategy="distance", max_hop=1)
            self.graph[m] = g
            As.append(torch.tensor(g.A, dtype=torch.float32, requires_grad=False))
            self.proj_linear[m] = nn.Linear(3, initial_gcn_dim)

        self.gcn_modules        = nn.ModuleDict()
        self.fusion_gcn_modules = nn.ModuleDict()

        final_dim_gcn = -1 
        for i, m in enumerate(self.modes):
            current_spatial_k = As[i].shape[0]
            gcn, d_mid = get_stgcn_chain(initial_gcn_dim, "spatial", (1, current_spatial_k), As[i].clone(), True)
            fus, d_out = get_stgcn_chain(d_mid, "temporal", (5, current_spatial_k), As[i].clone(), True)
            if i == 0: final_dim_gcn = d_out
            self.gcn_modules[m]        = gcn
            self.fusion_gcn_modules[m] = fus

        if "right" in self.modes and "left" in self.modes:
            self.gcn_modules["left"]        = self.gcn_modules["right"]
            self.fusion_gcn_modules["left"] = self.fusion_gcn_modules["right"]
            self.proj_linear["left"]        = self.proj_linear["right"]

        concat_dim    = final_dim_gcn * len(self.modes)
        self.part_para = nn.Parameter(torch.zeros(concat_dim))

        mt5_cfg           = MT5ForConditionalGeneration.from_pretrained(mt5_path).config
        self.mt5_model    = MT5ForConditionalGeneration.from_pretrained(mt5_path)
        
        # ==================================================================================
        # Prune mT5 layers if first_n_layer is specified
        n_layers = self.args.n_layers
        if n_layers is not None:
            n_layers = int(n_layers)
            if n_layers % 2 != 0:
                print(f"Warning: n_layers ({n_layers}) is not even. Pruning might not be perfectly symmetric.")
            
            half = n_layers // 2
            
            print(f"Pruning mT5: Keeping symmetric {n_layers} layers (First {half} & Last {half}) for Encoder and Decoder.")

            total_enc_blocks = len(self.mt5_model.encoder.block)
            if total_enc_blocks > n_layers:
                new_encoder_blocks = torch.nn.ModuleList([
                    self.mt5_model.encoder.block[i] for i in range(half)
                ] + [
                    self.mt5_model.encoder.block[i] for i in range(total_enc_blocks - half, total_enc_blocks)
                ])
                self.mt5_model.encoder.block = new_encoder_blocks

            total_dec_blocks = len(self.mt5_model.decoder.block)
            if total_dec_blocks > n_layers:
                new_decoder_blocks = torch.nn.ModuleList([
                    self.mt5_model.decoder.block[i] for i in range(half)
                ] + [
                    self.mt5_model.decoder.block[i] for i in range(total_dec_blocks - half, total_dec_blocks)
                ])
                self.mt5_model.decoder.block = new_decoder_blocks

            self.mt5_model.config.num_layers = len(self.mt5_model.encoder.block)
            self.mt5_model.config.num_decoder_layers = len(self.mt5_model.decoder.block)
            
            print(f"Pruning Complete. Final layers: Encoder={self.mt5_model.config.num_layers}, Decoder={self.mt5_model.config.num_decoder_layers}")
        # ==================================================================================

        self.mt5_tokenizer= T5Tokenizer.from_pretrained(mt5_path, legacy=False)
        self.mt5_dim      = mt5_cfg.d_model
        self.pose_proj    = nn.Linear(concat_dim, self.mt5_dim)
        self.enable_hyp_alignment = "hyperbolic" in self.args.loop.lower()

        if "CSL" in self.args.dataset:
            self.lang = 'Chinese'
        else:
            self.lang = 'English'
        
        if self.enable_hyp_alignment:
            self.hyp_dim    = args.hyp_dim

            if args.manifold == 'Euclidean':
                self.manifold   = Euclidean(ndim=1)
            elif args.manifold == 'PoincareBall':
                self.manifold   = PoincareBall(c=args.init_c, learnable=args.learnable_c)
            elif args.manifold == 'AdaptivePoincareBall':
                self.manifold = Scaled(PoincareBall(c=args.init_c, learnable=args.learnable_c), scale=args.hyp_scale, learnable=args.learnable_s)
            elif args.manifold == 'Lorentz':
                self.manifold = Lorentz(k=args.lorentz_k, learnable=True)
            elif args.manifold == 'AdaptiveLorentz':
                self.manifold = Scaled(Lorentz(k=args.lorentz_k, learnable=True), scale=args.hyp_scale, learnable=args.learnable_s)
            else:
                raise NotImplementedError
            
            self.hyp_proj_body  = HyperbolicMapper(final_dim_gcn, self.hyp_dim, self.manifold)
            self.hyp_proj_right = HyperbolicMapper(final_dim_gcn, self.hyp_dim, self.manifold)
            self.hyp_proj_left  = HyperbolicMapper(final_dim_gcn, self.hyp_dim, self.manifold)
            self.hyp_proj_face  = HyperbolicMapper(final_dim_gcn, self.hyp_dim, self.manifold)
            if self.args.manifold != 'Euclidean':
                self.hyp_proj_text  = HyperbolicMapper(self.mt5_dim,   self.hyp_dim, self.manifold)
            else:
                self.hyp_proj_text  = nn.Linear(self.mt5_dim, self.hyp_dim)

            self.hyp_attn_W = geoopt.ManifoldParameter(torch.randn(args.hyp_dim, args.hyp_dim), manifold=self.manifold)
            self.hyp_attn_b = geoopt.ManifoldParameter(torch.zeros(args.hyp_dim), manifold=self.manifold)

            self.geom_loss        = HyperbolicAlignmentLoss(args.label_smoothing_hyp, self.manifold, )
            self.loss_alpha_logit = nn.Parameter(torch.tensor(0.0))
            self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))
            self.total_steps      = max(int(getattr(args, 'total_steps', 1)), 1)
            self.text_cmp_mode    = getattr(args, "hyp_text_cmp", "pooled")
            self.hyp_text_emb_src = getattr(args, "hyp_text_emb_src", "decoder")

        self.apply(self._init_weights)

        self.enable_hyp_alignment = "hyperbolic" in self.args.loop.lower()
        print('Hyperbolic alignment is enable: ',  self.enable_hyp_alignment)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            if not isinstance(m.weight, ManifoldParameter):
                trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, src_input: Dict, tgt_input: Dict, call_from_kid=False) -> Dict[str, torch.Tensor]:
        if self.mt5_model is None or self.mt5_tokenizer is None:
            raise RuntimeError("mT5 model or tokenizer not loaded.")

        out, compute_dtype = {}, self.pose_proj.weight.dtype
        autocast_ctx = contextlib.nullcontext() 

        with autocast_ctx:
            feats, pooled, body_feat = [], {}, None
            active_modes = [m for m in self.modes if m in src_input]
            if not active_modes: raise ValueError("src_input contains no data for any defined modes.")

            for part in active_modes:
                x = self.proj_linear[part](src_input[part].to(dtype=compute_dtype)).permute(0,3,1,2)
                gcn_out = self.gcn_modules[part](x)
                if body_feat is not None:
                    if part == "left"   and body_feat.shape[-1] >= 2: gcn_out = gcn_out + body_feat[..., -2][..., None].detach()
                    elif part == "right" and body_feat.shape[-1] >= 1: gcn_out = gcn_out + body_feat[..., -1][..., None].detach()
                    elif part == "face_all" and body_feat.shape[-1] >= 1: gcn_out = gcn_out + body_feat[...,  0][..., None].detach()
                if part == "body": body_feat = gcn_out
                gcn_out = self.fusion_gcn_modules[part](gcn_out)
                if self.args.debug:
                    print('gcn_out', gcn_out.shape)
                pool_sp = gcn_out.mean(dim=-1).transpose(1,2)
                feats.append(pool_sp)
                pooled[part] = pool_sp.mean(dim=1)

            concatenated_feats = torch.cat(feats, dim=-1)
            pose_features_biased = concatenated_feats
            if len(active_modes) == len(self.modes): 
                pose_features_biased = concatenated_feats + self.part_para
            pose_emb = self.pose_proj(pose_features_biased)

            prompt_token = self.mt5_tokenizer(
                                    [f"Translate sign language video to {self.lang}: "] * len(tgt_input["gt_sentence"]),
                                    padding="longest",
                                    return_tensors="pt",
                                ).to(self.args.device)
            
            prompt_embeds = self.mt5_model.encoder.embed_tokens(prompt_token['input_ids'])
            inputs_embeds = torch.cat([prompt_embeds, pose_emb], dim=1)
            attention_mask = torch.cat([prompt_token['attention_mask'], src_input['attention_mask']], dim=1)

            tgt_input_tokenizer = self.mt5_tokenizer(tgt_input['gt_sentence'], 
                                                    return_tensors="pt", 
                                                    padding=True,
                                                    max_length=50)
                
            labels = tgt_input_tokenizer['input_ids']
            labels_masked = labels.clone()
            labels_masked[labels_masked == self.mt5_tokenizer.pad_token_id] = -100
            labels_masked = labels_masked.to(self.args.device)

            if "EcDec" in self.args.loop and self.args.num_loop > 0: 
                with torch.no_grad():
                    mt5_out = self.mt5_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                            labels=labels_masked, return_dict=True, output_hidden_states=True)
                    logits  = mt5_out.logits
                    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                            labels_masked.view(-1),
                                            label_smoothing=getattr(self.args, 'label_smoothing', 0.0),
                                            ignore_index=-100)
                    out["ce_loss"] = ce_loss.detach()
            else:
                mt5_out = self.mt5_model(inputs_embeds=inputs_embeds.to(self.args.device), attention_mask=attention_mask.to(self.args.device),
                                        labels=labels_masked.to(self.args.device), return_dict=True, output_hidden_states=True)
                logits  = mt5_out.logits
                ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                        labels_masked.view(-1),
                                        label_smoothing=getattr(self.args, 'label_smoothing', 0.0),
                                        ignore_index=-100)
                out["ce_loss"] = ce_loss.detach()

        if mt5_out.decoder_hidden_states is None:
            raise RuntimeError("Decoder hidden states not available for hyp_text_emb_src='decoder'.")
        out["last_decoder_hidden_states"] = mt5_out.decoder_hidden_states[-1]

        hyp_loss = torch.tensor(0.0, device=self.args.device)
        alpha_scalar = torch.tensor(self.args.hyp_alpha, device=self.args.device)
        geom_out     = {}
        current_step_eval_tensors_for_out = {}

        if self.enable_hyp_alignment:
            hyp_loss_ls = []
            with torch.amp.autocast("cuda", enabled=False):
                if not all(part in pooled for part in self.modes):
                    if self.args.eval and is_main_process(): 
                        warnings.warn(
                            "Skipping hyperbolic branch & eval_figure_data: Missing pooled features.",
                            stacklevel=1
                        )
                else:
                    if self.args.manifold != 'Lorentz':
                        norm = pooled["body"].norm(dim=-1, keepdim=True).clamp(min=1e-5)
                        pooled["body"] = pooled["body"] / norm
                        norm = pooled["left"].norm(dim=-1, keepdim=True).clamp(min=1e-5)
                        pooled["left"] = pooled["left"] / norm
                        norm = pooled["right"].norm(dim=-1, keepdim=True).clamp(min=1e-5)
                        pooled["right"] = pooled["right"] / norm
                        norm = pooled["face_all"].norm(dim=-1, keepdim=True).clamp(min=1e-5)
                        pooled["face_all"] = pooled["face_all"] / norm

                    if self.args.manifold != 'Euclidean':
                        hyp_body  = self.hyp_proj_body (pooled["body"].float())
                        hyp_left  = self.hyp_proj_left (pooled["left"].float())
                        hyp_right = self.hyp_proj_right(pooled["right"].float())
                        hyp_face  = self.hyp_proj_face (pooled["face_all"].float())
                        pose_points_stacked = torch.stack([hyp_body, hyp_left, hyp_right, hyp_face])
                    
                        if self.args.visualisation:
                            pass
                    else:
                        pose_points_stacked = torch.stack([pooled["body"], pooled["left"], pooled["right"], pooled["face_all"]])

                    if self.args.manifold != 'Euclidean':
                        d0s = torch.stack([self.manifold.dist0(p) for p in pose_points_stacked])
                    else:
                        d0s = torch.stack([p.norm(dim=-1) for p in pose_points_stacked])
                    w = torch.softmax(d0s, dim=0)
                    mu_mfd = compute_frechet_mean(pose_points_stacked, w, self.manifold)
                    
                    mask_bool = (labels != self.mt5_tokenizer.pad_token_id)
                    if mt5_out.decoder_hidden_states is None:
                        raise RuntimeError("Decoder hidden states not available for hyp_text_emb_src='decoder'.")
                    txt_e = mt5_out.decoder_hidden_states[-1]
                    mask_bool = mask_bool.to(txt_e.device)
                    
                    txt_mean = (txt_e * mask_bool.unsqueeze(-1).float()).sum(dim=1) / \
                                mask_bool.float().sum(dim=1, keepdim=True).clamp_min(1)
                    hyp_text_p = self.hyp_proj_text(txt_mean.float())
                    if self.args.visualisation:
                        pass
                    geom_out = self.geom_loss.forward(mu_mfd, hyp_text_p)
                    hyp_loss = geom_out["loss"]
                    hyp_loss_ls.append(geom_out["loss"])
                    
                    prog = self.global_step.item() / self.total_steps if self.total_steps > 0 else 0
                    a_base_val = getattr(self.args, 'alpha', 0.8)
                    a_base = a_base_val + (0.1 * prog)
                    a_learn = torch.sigmoid(self.loss_alpha_logit) * 0.2
                    alpha_scalar = (a_base + a_learn).clamp(0.1, 1.0) 
                    out.update({
                            "pose_points_stacked": pose_points_stacked,
                            "pose_frechet_mean": mu_mfd,
                            "hyp_text_p": hyp_text_p,
                            })
                    if geom_out:
                        log_weights_cond = 'w' in locals() and w is not None and w.numel() > 0
                        out.update({
                            "alpha": alpha_scalar.detach(),  
                            "hyp_sim_mean": geom_out.get("sim_mean", torch.tensor(0.0)),
                            "temperature": geom_out.get("temp", torch.tensor(0.0)),
                            "effective_margin": geom_out.get("margin", torch.tensor(0.0)),
                            "weights_fm_body" : w[0].mean().detach() if log_weights_cond and w.shape[0] > 0 else torch.tensor(0.0),
                            "weights_fm_left" : w[1].mean().detach() if log_weights_cond and w.shape[0] > 1 else torch.tensor(0.0),
                            "weights_fm_right": w[2].mean().detach() if log_weights_cond and w.shape[0] > 2 else torch.tensor(0.0),
                            "weights_fm_face" : w[3].mean().detach() if log_weights_cond and w.shape[0] > 3 else torch.tensor(0.0),
                            "hyp_loss": hyp_loss.detach(), 
                            # "hyp_loss_ls": hyp_loss_ls,
                            "alpha_hyp": alpha_scalar.detach(),      
                        })

        loss = ce_loss.float() 
        if self.enable_hyp_alignment and geom_out:
            loss = alpha_scalar * ce_loss.float() + (1 - alpha_scalar) * hyp_loss
        
        last_encoder_hidden_states = mt5_out.encoder_last_hidden_state.detach() if "EcDec" not in self.args.loop else None
        out.update({
            "loss": loss,
            "inputs_embeds": inputs_embeds.detach(),
            "last_encoder_hidden_states": last_encoder_hidden_states,
            "attention_mask": attention_mask.detach(),
            "labels":labels_masked
        })
        out["eval_figure_data"] = current_step_eval_tensors_for_out 
        return out

    @torch.no_grad()
    def generate(self, pc: Dict[str, torch.Tensor],
                 *, max_new_tokens: int = 100, num_beams: int = 4, call_from_kid=False, **kwargs) -> torch.Tensor:
        if not call_from_kid:
            if not {"inputs_embeds", "attention_mask"} <= pc.keys():
                if "body" in pc and "prefix_ids" in pc:
                    with torch.no_grad():
                        compute_dtype = self.pose_proj.weight.dtype
                        feats, _, body_feat = [], {}, None 
                        active_modes = [m for m in self.modes if m in pc]
                        if not active_modes: raise ValueError("Input pc contains no data for defined modes.")
                        for part in active_modes:
                            x = self.proj_linear[part](pc[part].to(dtype=compute_dtype)).permute(0,3,1,2)
                            gcn_out = self.gcn_modules[part](x)
                            if body_feat is not None:
                                if part == "left" and body_feat.shape[-1] >= 2: gcn_out += body_feat[..., -2][..., None].detach()
                                elif part == "right" and body_feat.shape[-1] >= 1: gcn_out += body_feat[..., -1][..., None].detach()
                                elif part == "face_all" and body_feat.shape[-1] >= 1: gcn_out += body_feat[..., 0][..., None].detach()
                            if part == "body": body_feat = gcn_out
                            gcn_out = self.fusion_gcn_modules[part](gcn_out)
                            pool_sp = gcn_out.mean(dim=-1).transpose(1,2)
                            feats.append(pool_sp)
                        concatenated_feats = torch.cat(feats, dim=-1)
                        pose_features_biased = concatenated_feats
                        if len(active_modes) == len(self.modes):
                            pose_features_biased += self.part_para
                        pose_emb = self.pose_proj(pose_features_biased)
                        prefix_ids    = pc["prefix_ids"].long()
                        prefix_mask   = pc["prefix_mask"]
                        if self.mt5_model is None: raise RuntimeError("mT5 model not loaded.")
                        inputs_embeds  = torch.cat([self.mt5_model.encoder.embed_tokens(prefix_ids), pose_emb], dim=1)
                        if "attention_mask" not in pc: raise ValueError("Pose attention_mask missing in pc for generation.")
                        attention_mask = torch.cat([prefix_mask, pc["attention_mask"]], dim=1)
                        pc_out = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
                else:
                    raise ValueError("generate: need 'inputs_embeds'/'attention_mask', or full src_input dict in pc.")
            else:
                pc_out = pc
            if self.mt5_model is None: raise RuntimeError("Cannot generate, mT5 model not loaded.")
            return self.mt5_model.generate(
                inputs_embeds  = pc_out["inputs_embeds"],
                attention_mask = pc_out["attention_mask"],
                max_new_tokens = max_new_tokens,
                num_beams      = num_beams,
                **kwargs
            )

def get_requires_grad_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    warnings.warn("get_requires_grad_dict is deprecated. Use model.state_dict().",
                  DeprecationWarning, stacklevel=2)
    param_req = {n: p.requires_grad for n, p in model.named_parameters()}
    for n,p in model.named_parameters():
        if isinstance(p, ManifoldParameter):
            warnings.warn(f"{n} is ManifoldParameter – may need geoopt to reload.", stacklevel=2)
    dup_map = {k.replace("left","right"):v for k,v in param_req.items() if "left" in k}
    param_req.update(dup_map)
    return {k:v for k,v in model.state_dict().items() if param_req.get(k, False)}