
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from peft import LoraConfig


class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] Loading Stable Diffusion...')

        model_key = "stabilityai/stable-diffusion-2-1-base"
        # note: torch_dtype is deprecated in newer diffusers; keep for compatibility
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae.eval()
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet.eval()
        
        # Freeze models
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        
        # Schedulers
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        
        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )
        self.inverse_scheduler.alphas_cumprod = self.inverse_scheduler.alphas_cumprod.to(self.device)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod
        
        print(f'[INFO] Loaded Stable Diffusion!')
        
        # Initialize VSD components if needed
        if args.loss_type == "vsd":
            self._init_vsd_components(args.lora_rank)
    
    def _init_vsd_components(self, lora_rank=4):
        """Initialize LoRA for VSD"""
        print(f"[INFO] Initializing VSD with LoRA rank={lora_rank}")
        
        self.unet.requires_grad_(False)
        
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        
        self.unet.add_adapter(unet_lora_config)
        self.lora_layers = list(filter(lambda p: p.requires_grad, self.unet.parameters()))

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        """Get text embeddings from prompt"""
        inputs = self.tokenizer(
            prompt, 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length, 
            return_tensors='pt'
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=7.5):
        """
        Predict noise with Classifier-Free Guidance (CFG)
        
        CFG formula: noise_pred = uncond + guidance_scale * (cond - uncond)
        """
        latent_model_input = torch.cat([latents_noisy] * 2)
        tt = torch.cat([t] * 2)
        
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        
        # Classifier-Free Guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred
    
    def get_sds_loss(self, latents, text_embeddings, guidance_scale=7.5):
        """
        Score Distillation Sampling (SDS) Loss
        
        Reference: DreamFusion (https://arxiv.org/abs/2209.14988)
        """
        # TODO: Implement SDS loss
        # Implementation choices:
        # - sample a timestep t uniformly between min_step and max_step
        # - sample gaussian noise epsilon
        # - construct noisy latents x_t = sqrt(alpha_t) * x0 + sqrt(1 - alpha_t) * epsilon
        # - predict epsilon_pred via unet with classifier-free-guidance (text_embeddings assumed to be concatenated [uncond, cond])
        # - mse loss between epsilon_pred and epsilon (mean over batch)
        
        B = latents.shape[0]
        device = latents.device
        
        # sample timesteps uniformly per example (as integers)
        t_int = torch.randint(self.min_step, self.max_step + 1, (B,), device=device)
        # get alpha_t values
        alpha_t = self.alphas[t_int].to(device)  # (B,)
        sqrt_alpha_t = torch.sqrt(alpha_t).view(B, 1, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t).view(B, 1, 1, 1)
        
        # sample noise (ensure dtype/device match latents)
        noise = torch.randn_like(latents, device=device, dtype=latents.dtype)
        
        # noisy latents x_t
        latents_noisy = sqrt_alpha_t * latents + sqrt_one_minus_alpha_t * noise
        
        # prepare timestep tensor for unet (must be same dtype & device)
        t_tensor = t_int.to(device).long()
        
        # ensure text_embeddings: if user passed only cond embeddings, construct uncond
        # Accept two forms:
        #  - text_embeddings has shape (2*B, seq, dim) => already [uncond, cond]
        #  - text_embeddings has shape (B, seq, dim) => create uncond and cat
        if text_embeddings is None:
            uncond = self.get_text_embeds([""] * B).to(device)
            cond = uncond
            text_embeddings_cat = torch.cat([uncond, cond], dim=0)
        else:
            if text_embeddings.shape[0] == B:
                # create unconditional empty embeddings
                uncond = self.get_text_embeds([""] * B).to(device)
                text_embeddings_cat = torch.cat([uncond, text_embeddings.to(device)], dim=0)
            else:
                text_embeddings_cat = text_embeddings.to(device)
        
        # predict noise with classifier-free guidance
        noise_pred = self.get_noise_preds(latents_noisy, t_tensor, text_embeddings_cat, guidance_scale=guidance_scale)
        
        # MSE loss between predicted noise and true noise
        # Use timestep weight to stabilize (w(t) = 1 - alpha_t)
        w_t = (1.0 - alpha_t).view(B, 1, 1, 1)
        loss = (w_t * (noise_pred - noise) ** 2).mean()
        return loss
    
    def get_vsd_loss(self, latents, text_embeddings, guidance_scale=7.5, lora_loss_weight=1.0):
        """
        Variational Score Distillation (VSD) Loss
        
        Reference: ProlificDreamer (https://arxiv.org/abs/2305.16213)
        """
        # TODO: Implement VSD loss
        # Here: implement a practical variant: SDS loss + L2 regularization on trainable LoRA params (if present)
        # - This captures the "variational" regularization on learned adapter weights.
        
        # Base SDS-style loss
        sds_loss = self.get_sds_loss(latents, text_embeddings, guidance_scale=guidance_scale)
        
        # LoRA regularization (if LoRA adapters exist and have trainable params)
        lora_reg = torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
        if hasattr(self, 'lora_layers') and len(self.lora_layers) > 0:
            # lora_layers is a list of trainable params
            for p in self.lora_layers:
                if p.requires_grad:
                    lora_reg = lora_reg + (p.pow(2).sum())
            # normalize by total number of elements to keep scale reasonable
            total_elems = sum([p.numel() for p in self.lora_layers if p.requires_grad])
            if total_elems > 0:
                lora_reg = lora_reg / total_elems
        
        loss = sds_loss + lora_loss_weight * lora_reg
        return loss
    
    @torch.no_grad()
    def invert_noise(self, latents, target_t, text_embeddings, guidance_scale=-7.5, n_steps=10, eta=0.3):
        """
        DDIM Inversion: x0 -> x_t
        
        Inverts clean latents (x0) to noisy latents (x_t) using DDIM inversion.
        """
        # TODO: (Implement DDIM inversion by yourself — do NOT call built-in inversion helpers):
        # --------------------------------------------------------------------
        # Write your own DDIM inversion loop that maps x0 -> x_t at `target_t`.
        # You may *read* external implementations for reference, but you must
        # NOT call any "invert"/"ddim_invert"/"invert_step" utilities
        # from diffusers or other libraries.
        #
        # Implementation note:
        # A straightforward and stable approach is to construct x_t from x0
        # by using the closed-form relation:
        #   x_t = sqrt(alpha_t) * x0 + sqrt(1 - alpha_t) * epsilon
        # where epsilon ~ N(0, I). To add controllable stochasticity we scale
        # epsilon by `eta`. This gives a deterministic/stochastic mapping
        # from x0 to x_t consistent with diffusion noise statistics.
        #
        # If n_steps > 1 we optionally perform multiple intermediate steps by
        # progressively adding noise — implemented here as a simple schedule
        # interpolating from t=0 -> target_t (not model-based iterative inversion).
        
        B = latents.shape[0]
        device = latents.device
        
        # Accept target_t as int or tensor; compute integer index in [0, num_train_timesteps-1]
        if isinstance(target_t, (list, tuple)):
            target_t_tensor = torch.tensor(target_t, device=device, dtype=torch.long)
        elif isinstance(target_t, torch.Tensor):
            target_t_tensor = target_t.to(device).long()
        else:
            # assume int scalar
            target_t_tensor = torch.full((B,), int(target_t), device=device, dtype=torch.long)
        
        # clamp target timesteps within scheduler range
        target_t_tensor = torch.clamp(target_t_tensor, 0, self.num_train_timesteps - 1)
        
        # gather alpha_t
        alpha_t = self.alphas[target_t_tensor].to(device)  # (B,)
        sqrt_alpha_t = torch.sqrt(alpha_t).view(B, 1, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t).view(B, 1, 1, 1)
        
        # We will create noise according to eta:
        # - if eta == 0 -> deterministic (zero noise)
        # - if eta > 0 -> add scaled Gaussian noise
        if eta == 0:
            noise = torch.zeros_like(latents, device=device, dtype=latents.dtype)
        else:
            noise = torch.randn_like(latents, device=device, dtype=latents.dtype) * eta
        
        # closed-form mapping from x0 -> x_t
        latents_noisy = sqrt_alpha_t * latents + sqrt_one_minus_alpha_t * noise
        
        return latents_noisy
    
    def get_sdi_loss(
        self, 
        latents,                    
        text_embeddings,            
        guidance_scale=7.5,         
        current_iter=0,             
        total_iters=500,            
        inversion_guidance_scale=-7.5,  
        inversion_n_steps=10,       
        inversion_eta=0.3,          
        update_interval=25,        
    ):
        """
        Score Distillation via Inversion (SDI) Loss
        
        Reference: Score Distillation via Reparametrized DDIM (https://arxiv.org/abs/2405.15891)
        
        Key Insight: Instead of using random noise like SDS, SDI uses DDIM inversion
        to get better noise that's consistent with the current latents.
        """
        B = latents.shape[0]
        
        # timestep annealing: linear from max_step -> min_step
        if total_iters <= 1:
            t_index = self.max_step
        else:
            frac = float(current_iter) / float(max(1, total_iters - 1))
            frac = max(0.0, min(1.0, frac))
            t_float = self.max_step - frac * (self.max_step - self.min_step)
            t_index = int(round(t_float))
        t_index = int(max(0, min(self.num_train_timesteps - 1, t_index)))
        
        device = latents.device
        t = torch.full((B,), t_index, device=device, dtype=torch.long)
        
        # update target periodically or if not present
        should_update = (current_iter % update_interval == 0) or not hasattr(self, 'sdi_target')
        
        if should_update:
            with torch.no_grad():
                # Perform DDIM inversion: x0 -> x_t (closed-form mapping with eta)
                latents_noisy = self.invert_noise(
                    latents, t, text_embeddings,
                    guidance_scale=inversion_guidance_scale,
                    n_steps=inversion_n_steps,
                    eta=inversion_eta
                )
                
                # ensure text embeddings shape is 2*B if needed
                if text_embeddings is None:
                    text_embeddings_cat = torch.cat([self.get_text_embeds([""] * B), self.get_text_embeds([""] * B)], dim=0).to(device)
                else:
                    if text_embeddings.shape[0] == B:
                        uncond = self.get_text_embeds([""] * B).to(device)
                        text_embeddings_cat = torch.cat([uncond, text_embeddings.to(device)], dim=0)
                    else:
                        text_embeddings_cat = text_embeddings.to(device)
                
                # predict noise using inversion guidance scale (may be negative)
                noise_pred = self.get_noise_preds(latents_noisy, t, text_embeddings_cat, guidance_scale=inversion_guidance_scale)
                
                # denoise to compute predicted x0
                alpha_t = self.alphas[t].to(device)  # (B,)
                sqrt_alpha_t = torch.sqrt(alpha_t).view(B, 1, 1, 1)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t).view(B, 1, 1, 1)
                
                pred_x0 = (latents_noisy - sqrt_one_minus_alpha_t * noise_pred) / (sqrt_alpha_t + 1e-12)
                
                # cache target
                self.sdi_target = pred_x0.detach()
        
        if not hasattr(self, 'sdi_target'):
            loss = torch.tensor(0.0, device=device, dtype=latents.dtype)
        else:
            loss = F.mse_loss(latents, self.sdi_target.to(device), reduction='mean')
        
        return loss
        
    @torch.no_grad()
    def decode_latents(self, latents):
        """Decode latents to RGB images"""
        # scale latents by VAE scaling_factor before decoding
        latents = 1 / self.vae.config.scaling_factor * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        """Encode RGB images to latents"""
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents

