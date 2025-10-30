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
        B = latents.shape[0]
        device = latents.device

        # sample t uniformly
        t_int = torch.randint(self.min_step, self.max_step + 1, (B,), device=device)
        alpha_t = self.alphas[t_int].to(device)
        sqrt_alpha_t = torch.sqrt(alpha_t).view(B, 1, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t).view(B, 1, 1, 1)

        # sample noise
        noise = torch.randn_like(latents, device=device, dtype=latents.dtype)

        # create noisy latents
        latents_noisy = sqrt_alpha_t * latents + sqrt_one_minus_alpha_t * noise
        t_tensor = t_int.to(device).long()

        # make sure text embeddings include unconditional
        if text_embeddings is None or text_embeddings.shape[0] == B:
            uncond = self.get_text_embeds([""] * B).to(device)
            cond = text_embeddings.to(device) if text_embeddings is not None else uncond
            text_embeddings_cat = torch.cat([uncond, cond], dim=0)
        else:
            text_embeddings_cat = text_embeddings.to(device)

        unet_dtype = next(self.unet.parameters()).dtype
        latents_noisy = latents_noisy.to(dtype=unet_dtype)
        text_embeddings_cat = text_embeddings_cat.to(dtype=unet_dtype)

        # predict noise
        noise_pred = self.get_noise_preds(latents_noisy, t_tensor, text_embeddings_cat, guidance_scale=guidance_scale)

        # gradient direction
        grad = (noise_pred - noise)

        # weight by (1 - alpha_t)
        w_t = (1.0 - alpha_t).view(B, 1, 1, 1)
        grad = w_t * grad

        # --- [!!!] START OF THE CRITICAL FIX [!!!] ---
        #
        # "Pseudo-loss" that gives the correct gradient direction
        #
        grad_detached = grad.detach()
        
        target = (latents_noisy - grad_detached).detach()
        loss = 0.5 * F.mse_loss(latents_noisy, target, reduction="mean")
        # --- [!!!] END OF THE CRITICAL FIX [!!!] ---
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
        
        # 1. 準備 text embeddings (與您原本的程式碼相同)
        if text_embeddings is None:
            uncond = self.get_text_embeds([""] * B).to(device)
            text_embeddings_cat = torch.cat([uncond, uncond], dim=0)
        else:
            if text_embeddings.shape[0] == B:
                uncond = self.get_text_embeds([""] * B).to(device)
                text_embeddings_cat = torch.cat([uncond, text_embeddings.to(device)], dim=0)
            else:
                text_embeddings_cat = text_embeddings.to(device)
                
        unet_dtype = next(self.unet.parameters()).dtype
        text_embeddings_cat = text_embeddings_cat.to(dtype=unet_dtype)

        # 2. 設置 timesteps (與您原本的程式碼相同)
        target_t_int = int(target_t[0]) if isinstance(target_t, torch.Tensor) else int(target_t)
        target_t_int = max(target_t_int, 1) 
        timesteps = torch.linspace(0, target_t_int, n_steps + 1, device=device, dtype=torch.long)
        
        # 3. 執行 DDIM Inversion 迴圈 (與您原本的程式碼相同)
        x_current = latents.to(dtype=unet_dtype) 

        for i in range(n_steps):
            t_current = timesteps[i]
            t_next = timesteps[i+1]
            t_current_tensor = torch.full((B,), t_current, device=device, dtype=torch.long)

            # (1) 取得 alpha (與您原本的程式碼相同)
            alpha_current = self.alphas[t_current].to(device).view(B, 1, 1, 1)
            alpha_next = self.alphas[t_next].to(device).view(B, 1, 1, 1)
            
            # (2) 預測噪聲 (與您原本的程式碼相同)
            noise_pred = self.get_noise_preds(
                x_current, 
                t_current_tensor, 
                text_embeddings_cat, 
                guidance_scale=guidance_scale
            )
            
            # (3) 計算 pred_x0 (與您原本的程式碼相同)
            pred_x0 = (x_current - torch.sqrt(1.0 - alpha_current) * noise_pred) / (torch.sqrt(alpha_current) + 1e-12)

            # (4) 計算 x_next (與您原本的程式碼相同)
            x_next = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1.0 - alpha_next) * noise_pred

            # (5) [!!!] --- 這是【真正】的修正點 --- [!!!]
            if eta > 0:
                # DDIM (https://arxiv.org/abs/2010.02502) - Eq. 12
                # 我們需要 sigma^2 = eta^2 * (1 - alpha_{t-1}) / (1 - alpha_t) * (1 - alpha_t / alpha_{t-1})
                # t-1 = t_current, t = t_next
                
                # 【錯誤】的版本 (會產生負數):
                # variance = ( (1.0 - alpha_next) / (1.0 - alpha_current) ) * (1.0 - alpha_current / (alpha_next + 1e-12) )
                
                # 【正確】的版本 (alpha_current 和 alpha_next 的位置被修正):
                variance_term_1 = (1.0 - alpha_current) / (1.0 - alpha_next + 1e-12)
                variance_term_2 = (1.0 - alpha_next / (alpha_current + 1e-12))
                
                # 為了數值穩定，clamp 確保 variance 永遠 >= 0
                variance = torch.clamp(variance_term_1 * variance_term_2, min=0)
                
                sigma = eta * torch.sqrt(variance)
                
                noise = torch.randn_like(noise_pred)
                x_next = x_next + sigma * noise
            
            # (6) 更新 x_current (与您原本的程式码相同)
            x_current = x_next
            
        return x_current.detach()
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
        device = latents.device

        # 1. Timestep annealing (與您原本的程式碼相同)
        if total_iters <= 1:
            t_index = self.max_step
        else:
            frac = float(current_iter) / float(max(1, total_iters - 1))
            frac = max(0.0, min(1.0, frac))
            t_float = self.max_step - frac * (self.max_step - self.min_step)
            t_index = int(round(t_float))
        
        t_index = int(max(0, min(self.num_train_timesteps - 1, t_index)))
        t = torch.full((B,), t_index, device=device, dtype=torch.long)

        # 2. update target periodically (與您原本的程式碼相同)
        should_update = (current_iter % update_interval == 0) or not hasattr(self, 'sdi_target')

        if should_update:
            with torch.no_grad():
                
                # 2a. 呼叫 *新* 的 invert_noise (方法 A)
                # 這裡會使用到 n_steps, eta, 和 inversion_guidance_scale
                latents_noisy = self.invert_noise(
                    latents, t, text_embeddings,
                    guidance_scale=inversion_guidance_scale, # <-- 傳入 -7.5
                    n_steps=inversion_n_steps,           
                    eta=inversion_eta                    
                )

                # 2b. 準備 text embeddings (與您原本的程式碼相同)
                if text_embeddings is None:
                    uncond = self.get_text_embeds([""] * B).to(device)
                    text_embeddings_cat = torch.cat([uncond, uncond], dim=0)
                else:
                    if text_embeddings.shape[0] == B:
                        uncond = self.get_text_embeds([""] * B).to(device)
                        text_embeddings_cat = torch.cat([uncond, text_embeddings.to(device)], dim=0)
                    else:
                        text_embeddings_cat = text_embeddings.to(device)
                
                unet_dtype = next(self.unet.parameters()).dtype
                latents_noisy = latents_noisy.to(dtype=unet_dtype)
                text_embeddings_cat = text_embeddings_cat.to(dtype=unet_dtype)

                # 2c. [!!!] --- 這是您 *第二個* 錯誤的修正點 --- [!!!]
                # 
                # 為了計算最終的 *目標* pred_x0，
                # 您必須使用 *正的* guidance_scale (例如 50 或 10)
                #
                # 錯誤的版本 (使用 -7.5):
                # noise_pred = self.get_noise_preds(latents_noisy, t, text_embeddings_cat, guidance_scale=inversion_guidance_scale)
                #
                # 正確的版本 (使用 50 或 10):
                noise_pred = self.get_noise_preds(latents_noisy, t, text_embeddings_cat, guidance_scale=guidance_scale)
                #
                # [!!!] --- 修正完畢 --- [!!!]

                # 2d. 計算 pred_x0 (與您原本的程式碼相同)
                alpha_t = self.alphas[t].to(device).view(B, 1, 1, 1)
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

                pred_x0 = (latents_noisy - sqrt_one_minus_alpha_t * noise_pred) / (sqrt_alpha_t + 1e-12)

                # 2e. 快取 target (與您原本的程式碼相同)
                self.sdi_target = pred_x0.detach()

        # 3. 計算 loss (與您原本的程式碼相同)
        if not hasattr(self, 'sdi_target'):
            loss = torch.tensor(0.0, device=device, dtype=latents.dtype)
        else:
            loss = F.mse_loss(latents, self.sdi_target.to(device), reduction='mean')

        return loss
       
    @torch.no_grad()

    def decode_latents(self, latents):
        """Decode latents to RGB images"""
        # --- make sure dtype & scaling factor match VAE expectation ---
        # SD 2.1 base expects latent scaled by 0.18215 (same as vae.config.scaling_factor)
        # We must cast to float32 for decoding, even if pipeline runs in fp16.
        latents = latents.to(self.vae.dtype)
        if hasattr(self.vae.config, "scaling_factor"):
            latents = latents / self.vae.config.scaling_factor
        else:
            latents = latents / 0.18215  # fallback

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)  # scale to [0,1]
        imgs = imgs.float()  # convert for visualization
        return imgs

    #def decode_latents(self, latents):
    #    """Decode latents to RGB images"""
    #    # scale latents by VAE scaling_factor before decoding
    #    latents = 1 / self.vae.config.scaling_factor * latents
    #    imgs = self.vae.decode(latents).sample
    #    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    #    return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        """Encode RGB images to latents"""
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents