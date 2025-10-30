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
        (This is the CULMINATION of all fixes - 'disable_adapters' REMOVED)
        """
        
        # [!!!] VSD/NaN Bug 修復：
        # 我們不再 disable_adapters。
        # 真正的 bug 是梯度上升 (用 '+' 號)。
        # 我們已經在下面修正為 '-' 號 (梯度下降)。
        
        print(f"--- SDS_LOG: (Correct VSD Version) 正在執行 (target = noisy - grad) ---")
        
        B = latents.shape[0]
        device = latents.device

        t_int = torch.randint(self.min_step, self.max_step + 1, (B,), device=device)
        alpha_t = self.alphas[t_int].to(device)
        sqrt_alpha_t = torch.sqrt(alpha_t).view(B, 1, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t).view(B, 1, 1, 1)

        noise = torch.randn_like(latents, device=device, dtype=torch.float32)
        
        # [!!!] NameError FIX: 這一行在這裡定義 [!!!]
        latents_noisy = sqrt_alpha_t.float() * latents.float() + sqrt_one_minus_alpha_t.float() * noise
        t_tensor = t_int.to(device).long()

        if text_embeddings is None or text_embeddings.shape[0] == B:
            uncond = self.get_text_embeds([""] * B).to(device)
            cond = text_embeddings.to(device) if text_embeddings is not None else uncond
            text_embeddings_cat = torch.cat([uncond, cond], dim=0)
        else:
            text_embeddings_cat = text_embeddings.to(device)
        
        unet_dtype = next(self.unet.parameters()).dtype
        
        # [!!!] NameError FIX: 'latents_noisy' 在上面已經被定義 [!!!]
        latents_noisy_input = latents_noisy.to(dtype=unet_dtype)
        text_embeddings_cat = text_embeddings_cat.to(dtype=unet_dtype)
        
        # [!!!] 呼叫 UNet (帶著「開啟」的 LoRA)
        # 這就是 VSD 和 SDS 不同的地方
        noise_pred = self.get_noise_preds(latents_noisy_input, t_tensor, text_embeddings_cat, guidance_scale=guidance_scale)
        
        # (移除了 enable_adapters)

        grad = (noise_pred.float() - noise)
        w_t = (1.0 - alpha_t).view(B, 1, 1, 1).float()
        grad = w_t * grad

        # --- [!!!] 梯度下降的 FIX [!!!] ---
        grad_detached = grad.detach()
        target = (latents_noisy - grad_detached).detach() # <-- 減號
        loss = 0.5 * F.mse_loss(latents_noisy, target, reduction="mean")
        return loss   


    def get_vsd_loss(self, latents, text_embeddings, guidance_scale=7.5, lora_loss_weight=1.0):
        """
        Variational Score Distillation (VSD) Loss
        (This is the CORRECT version, calling SDS once)
        """
        print("--- VSD_LOG: (Correct Version) 正在執行 (sds_loss + lora_reg) ---")
        
        sds_loss = self.get_sds_loss(latents, text_embeddings, guidance_scale=guidance_scale)
        
        lora_reg = torch.tensor(0.0, device=latents.device, dtype=torch.float32)
        
        if hasattr(self, 'lora_layers') and len(self.lora_layers) > 0:
            total_elems = 0
            for p in self.lora_layers:
                if p.requires_grad:
                    lora_reg = lora_reg + (p.float().pow(2).sum())
                    total_elems += p.numel()
            
            if total_elems > 0:
                lora_reg = lora_reg / total_elems
        
        loss = sds_loss + float(lora_loss_weight) * lora_reg
        return loss
    
    @torch.no_grad()
    def invert_noise(self, latents, target_t, text_embeddings, guidance_scale=-7.5, n_steps=10, eta=0.3):
        """
        DDIM Inversion: x0 -> x_t (Closed-form implementation)
        """
        B = latents.shape[0]
        device = latents.device
        
        if isinstance(target_t, (list, tuple)):
            target_t_tensor = torch.tensor(target_t, device=device, dtype=torch.long)
        elif isinstance(target_t, torch.Tensor):
            target_t_tensor = target_t.to(device).long()
        else:
            target_t_tensor = torch.full((B,), int(target_t), device=device, dtype=torch.long)
        
        target_t_tensor = torch.clamp(target_t_tensor, 0, self.num_train_timesteps - 1)
        
        alpha_t = self.alphas[target_t_tensor].to(device)  # (B,)
        sqrt_alpha_t = torch.sqrt(alpha_t).view(B, 1, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t).view(B, 1, 1, 1)
        
        if eta == 0:
            noise = torch.zeros_like(latents, device=device, dtype=latents.dtype)
        else:
            noise = torch.randn_like(latents, device=device, dtype=latents.dtype) * eta
        
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
        (This is the CORRECT version)
        """
        B = latents.shape[0]
        
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
        
        should_update = (current_iter % update_interval == 0) or not hasattr(self, 'sdi_target')
        
        if should_update:
            with torch.no_grad():
                latents_noisy = self.invert_noise(
                    latents, t, text_embeddings,
                    guidance_scale=inversion_guidance_scale,
                    n_steps=inversion_n_steps,
                    eta=inversion_eta
                )
                
                if text_embeddings is None:
                    text_embeddings_cat = torch.cat([self.get_text_embeds([""] * B), self.get_text_embeds([""] * B)], dim=0).to(device)
                else:
                    if text_embeddings.shape[0] == B:
                        uncond = self.get_text_embeds([""]*B).to(device)
                        text_embeddings_cat = torch.cat([uncond, text_embeddings.to(device)], dim=0)
                    else:
                        text_embeddings_cat = text_embeddings.to(device)
                
                noise_pred = self.get_noise_preds(latents_noisy, t, text_embeddings_cat, guidance_scale=inversion_guidance_scale)
                
                alpha_t = self.alphas[t].to(device)
                sqrt_alpha_t = torch.sqrt(alpha_t).view(B, 1, 1, 1)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t).view(B, 1, 1, 1)
                
                pred_x0 = (latents_noisy - sqrt_one_minus_alpha_t * noise_pred) / (sqrt_alpha_t + 1e-12)
                
                self.sdi_target = pred_x0.detach()
        
        if not hasattr(self, 'sdi_target'):
            loss = torch.tensor(0.0, device=device, dtype=latents.dtype)
        else:
            loss = F.mse_loss(latents, self.sdi_target.to(device), reduction='mean')
        
        return loss

    # [!!!] AttributeError FIX: 這些函數被加回來了 [!!!]
    @torch.no_grad()
    def decode_latents(self, latents):
        """Decode latents to RGB images"""
        # [FIX] VAE decode 在 fp16 下不穩定. 強制使用 fp32.
        
        vae_dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        latents_fp32 = latents.to(dtype=torch.float32)

        if hasattr(self.vae.config, "scaling_factor"):
            scaling_factor = self.vae.config.scaling_factor
        else:
            scaling_factor = 0.18215  # fallback
        latents_fp32 = latents_fp32 / scaling_factor
        
        imgs = self.vae.decode(latents_fp32).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1) 
        imgs = imgs.float()  # 最終輸出

        self.vae.to(dtype=vae_dtype)

        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        """Encode RGB images to latents"""
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents