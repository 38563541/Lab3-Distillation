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
        
        # sample noise
        noise = torch.randn_like(latents, device=latents.device, dtype=latents.dtype)

        # noisy latents x_t
        latents_noisy = sqrt_alpha_t * latents + sqrt_one_minus_alpha_t * noise
        
        # prepare timestep tensor for unet (must be same dtype & device)
        t_tensor = t_int.to(device).long()
        
        # ensure text_embeddings: if user passed only cond embeddings, construct uncond
        if text_embeddings.shape[0] == B:
            uncond = self.get_text_embeds([""] * B).to(device)
            text_embeddings_cat = torch.cat([uncond, text_embeddings], dim=0)
