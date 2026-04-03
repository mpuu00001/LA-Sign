import torch
from torch import nn
import torch.utils.checkpoint
import contextlib 

from models.tools import sample_poisson_lognormal
from models.hyperbolic import Uni_Sign_Hyperbolic
        
class Sign_Loop_Hyperbolic(Uni_Sign_Hyperbolic):
    def __init__(self, args):
        super().__init__(args)
        print(f"Sign_Loop_Hyperbolic initialized with mode: {args.loop}")
        
        self.loop_type = args.loop  
        
        if self.enable_hyp_alignment:
            self.hyp_pose_proj = nn.Linear(self.args.hyp_dim, args.dim)
        
        self.all_layer_align = args.all_layer_align
        self.aux_loss_weight = args.aux_loss_weight

    @contextlib.contextmanager
    def freeze_decoder_temporarily(self):
        modules_to_freeze = [self.mt5_model.decoder.block, self.mt5_model.decoder.final_layer_norm]
        if hasattr(self.mt5_model, 'lm_head'):
            modules_to_freeze.append(self.mt5_model.lm_head)

        params_to_restore = []
        for module in modules_to_freeze:
            for param in module.parameters():
                if param.requires_grad:
                    params_to_restore.append(param)
                    param.requires_grad = False
        try:
            yield
        finally:
            for param in params_to_restore:
                param.requires_grad = True

    def compute_step_hyp_loss(self, pose_hyp_mean, text_features, labels):
        mask_bool = (labels != self.mt5_tokenizer.pad_token_id).to(text_features.device)
        
        denom = mask_bool.float().sum(dim=1, keepdim=True).clamp_min(1)
        txt_mean = (text_features * mask_bool.unsqueeze(-1).float()).sum(dim=1) / denom
        
        hyp_text_p = self.hyp_proj_text(txt_mean.float())
        
        geom_out = self.geom_loss.forward(pose_hyp_mean, hyp_text_p)
        
        return geom_out["loss"]

    def _loop_ecdec(self, inputs_embeds, attention_mask, start_idx, labels=None, pose_hyp_mean=None, aux_losses=None, is_training=False):

        num_loop = max(1, sample_poisson_lognormal(target_mean=self.args.distr_mean)) if (self.args.use_dynamic_loops and is_training) else self.args.num_loop
        loop_range = range(start_idx, int(num_loop))
        
        init_inputs_embeds = inputs_embeds
        init_atten_mask = attention_mask

        for i, _ in enumerate(loop_range):
            if is_training:
                out = self.mt5_model(
                    inputs_embeds=inputs_embeds,
                    labels=labels.to(inputs_embeds.device),
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                
                is_last_layer = (i == len(loop_range) - 1)
                if self.all_layer_align and self.enable_hyp_alignment and (not is_last_layer) and (pose_hyp_mean is not None):
                    step_loss = self.compute_step_hyp_loss(pose_hyp_mean, out.decoder_hidden_states[-1], labels)
                    aux_losses.append(step_loss)

            last_decoder_state = out.decoder_hidden_states[-1]
            inputs_embeds = torch.cat([init_inputs_embeds, last_decoder_state], dim=1)
            new_mask = torch.ones(attention_mask.size(0), last_decoder_state.size(1), 
                                  dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([init_atten_mask, new_mask], dim=1)

        return out, aux_losses

    def _loop_ec(self, inputs_embeds, attention_mask, start_idx,labels=None, pose_hyp_mean=None, aux_losses=None, is_training=False):

        init_inputs_embeds = inputs_embeds
        num_loop = max(1, sample_poisson_lognormal(target_mean=self.args.distr_mean)) if (self.args.use_dynamic_loops and is_training) else self.args.num_loop
        loop_range = range(start_idx, int(num_loop))

        for i, _ in enumerate(loop_range):
            encoder_outputs = self.mt5_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            if self.all_layer_align and self.enable_hyp_alignment and (pose_hyp_mean is not None) and (labels is not None):
                with self.freeze_decoder_temporarily():
                    curr_labels = labels.to(inputs_embeds.device)
                    decoder_out = self.mt5_model(
                        encoder_outputs=encoder_outputs,
                        attention_mask=attention_mask,
                        labels=curr_labels,
                        return_dict=True,
                        output_hidden_states=True
                    )
                    step_loss = self.compute_step_hyp_loss(pose_hyp_mean, decoder_out.decoder_hidden_states[-1], curr_labels)
                    aux_losses.append(step_loss)

            last_encoder_state = encoder_outputs[0]
            inputs_embeds = init_inputs_embeds + last_encoder_state

        return inputs_embeds, attention_mask, aux_losses

    def _loop_dec(self, decoder_input_embeds, encoder_hidden_states, encoder_attention_mask, start_idx, labels=None, pose_hyp_mean=None, aux_losses=None, is_training=False):
        inputs_embeds = decoder_input_embeds
        loop_range = range(start_idx, self.args.num_loop)

        num_loop = max(1, sample_poisson_lognormal(target_mean=self.args.distr_mean)) if (self.args.use_dynamic_loops and is_training) else self.args.num_loop
        loop_range = range(start_idx, int(num_loop))

        decoder_outputs = None
        for i, _ in enumerate(loop_range):
            decoder_outputs = self.mt5_model.decoder(
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=True,
                output_hidden_states=True            
            )
            
            last_decoder_state = decoder_outputs.last_hidden_state
            
            is_last_layer = (i == len(loop_range) - 1)
            if self.all_layer_align and self.enable_hyp_alignment and (not is_last_layer) and (pose_hyp_mean is not None) and (labels is not None):
                step_loss = self.compute_step_hyp_loss(pose_hyp_mean, last_decoder_state, labels)
                aux_losses.append(step_loss)

            inputs_embeds = last_decoder_state

        return decoder_outputs, aux_losses


    def forward(self, src_input, tgt_input, call_from_kid=True):
        base_out = super().forward(src_input, tgt_input, call_from_kid=call_from_kid)
        
        inputs_embeds = base_out['inputs_embeds']
        attention_mask = base_out['attention_mask']
        labels = base_out['labels']
        pose_hyp_mean = base_out.get('pose_frechet_mean')

        final_out_for_loss = None  
        text_feature_for_final_hyp = None 
        aux_losses = []

        if self.args.num_loop > 0:
            if "EcDec" in self.loop_type:
                # Loop Encoder-decoder
                out, aux_losses = self._loop_ecdec(
                    inputs_embeds, attention_mask, labels=labels,
                    start_idx=0, pose_hyp_mean=pose_hyp_mean, 
                    aux_losses=aux_losses, is_training=True
                )
                final_out_for_loss = out
                text_feature_for_final_hyp = out.decoder_hidden_states[-1]

            elif "SignEc" in self.loop_type:
                last_enc_states = base_out['last_encoder_hidden_states']
                inputs_embeds = torch.cat([inputs_embeds, last_enc_states], dim=1)
                new_mask = torch.ones(attention_mask.size(0), last_enc_states.size(1), 
                                      dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, new_mask], dim=1)

                # Loop Encoder
                inputs_embeds, attention_mask, aux_losses = self._loop_ec(
                    inputs_embeds, attention_mask, labels=labels,
                    start_idx=1, pose_hyp_mean=pose_hyp_mean, aux_losses=aux_losses, is_training=True
                )

                # Final Decoder Pass
                out = self.mt5_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels.to(inputs_embeds.device),
                    return_dict=True,
                    output_hidden_states=True
                )
                final_out_for_loss = out
                text_feature_for_final_hyp = out.decoder_hidden_states[-1]

            elif "SignDec" in self.loop_type:
                encoder_hidden_states = base_out['last_encoder_hidden_states']
                decoder_input_embeds = base_out['last_decoder_hidden_states']
                
                # Loop Decoder
                decoder_outputs, aux_losses = self._loop_dec(
                    decoder_input_embeds, encoder_hidden_states, attention_mask,
                    start_idx=0, labels=labels, pose_hyp_mean=pose_hyp_mean, aux_losses=aux_losses, is_training=True
                )
                
                # Manual LM Head
                sequence_output = decoder_outputs.last_hidden_state
                lm_logits = self.mt5_model.lm_head(sequence_output)
                
                # Mock an output object for consistency
                final_out_for_loss = type('obj', (object,), {'logits': lm_logits})
                text_feature_for_final_hyp = sequence_output

        else:
            final_loss = base_out['loss']
            lm_loss = base_out.get('ce_loss', final_loss)
            hyp_loss = base_out['hyp_loss']
            stack_out = {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask,
                'loss': final_loss,
                'lm_loss': lm_loss,
                'hyp_loss': hyp_loss
            }
            return stack_out

        label = labels.reshape(-1)
        out_logits = final_out_for_loss.logits
        logits = out_logits.reshape(-1, out_logits.shape[-1])
        loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing, ignore_index=-100)
        lm_loss = loss_fct(logits, label.to(out_logits.device, non_blocking=True))

        if self.enable_hyp_alignment and (pose_hyp_mean is not None):
            final_hyp_loss = self.compute_step_hyp_loss(pose_hyp_mean, text_feature_for_final_hyp, labels)
            
            if self.all_layer_align and len(aux_losses) > 0:
                sum_aux_loss = torch.stack(aux_losses).sum()
                total_hyp_loss = final_hyp_loss + (self.aux_loss_weight * sum_aux_loss)
            else:
                total_hyp_loss = final_hyp_loss
            
            base_out['margin_loss'] = total_hyp_loss
            
            alpha_scalar = base_out['alpha']
            final_loss = alpha_scalar * lm_loss.float() + (1 - alpha_scalar) * total_hyp_loss
        else:
            final_loss = lm_loss

        stack_out = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'loss': final_loss,
            'lm_loss': lm_loss,
            'hyp_loss': total_hyp_loss
        }
        return stack_out

    @torch.no_grad()
    def generate(self, pre_compute_item, max_new_tokens, num_beams, scr_lens=None, synced_gpus=False):
        inputs_embeds = pre_compute_item['inputs_embeds']
        attention_mask = pre_compute_item['attention_mask']

        decoder_input_ids = torch.full(
            (inputs_embeds.size(0), 1),
            self.mt5_model.config.decoder_start_token_id,
            dtype=torch.long,
            device=self.args.device,
        )

        if self.args.num_loop > 0:
            
            if "EcDec" in self.loop_type:
                init_inputs_embeds = inputs_embeds
                init_atten_mask = attention_mask
                
                for _ in range(self.args.num_loop):
                    out = self.mt5_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    last_dec = out.decoder_hidden_states[-1]
                    inputs_embeds = torch.cat([init_inputs_embeds, last_dec], dim=1)
                    new_mask = torch.ones(attention_mask.size(0), last_dec.size(1), 
                                          dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat([init_atten_mask, new_mask], dim=1)

                return self.mt5_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    synced_gpus=synced_gpus
                )

            elif "SignEc" in self.loop_type:
                encoder_outputs = self.mt5_model.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
                last_enc = encoder_outputs.last_hidden_state
                
                inputs_embeds = torch.cat([inputs_embeds, last_enc], dim=1)
                new_mask = torch.ones(attention_mask.size(0), last_enc.size(1), 
                                      dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, new_mask], dim=1)
                
                inputs_embeds, attention_mask, _ = self._loop_ec(
                    inputs_embeds, attention_mask, start_idx=1
                )
                
                return self.mt5_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    synced_gpus=synced_gpus
                )

            elif "SignDec" in self.loop_type:
                encoder_outputs = self.mt5_model.encoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                enc_hidden = encoder_outputs.last_hidden_state
                
                dec_embeds = self.mt5_model.decoder.embed_tokens(decoder_input_ids)
                
                dec_outputs, _ = self._loop_dec(
                    dec_embeds, enc_hidden, attention_mask, start_idx=0 
                )
                
                final_dec_input_embeds = dec_outputs.last_hidden_state
                
                return self.mt5_model.generate(
                    input_ids=None,
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    decoder_inputs_embeds=final_dec_input_embeds,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    synced_gpus=synced_gpus
                )

        else:
            return self.mt5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                synced_gpus=synced_gpus
            )
        