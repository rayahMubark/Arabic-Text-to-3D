# Arabic-Aware Conditioning
from triposg.text.arabic_conditioning import ArabicAwareConditioner
from transformers import CLIPTextModel, CLIPTokenizer

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import PIL.Image
import torch
import trimesh
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    BitImageProcessor,
    Dinov2Model,
)
from ..inference_utils import hierarchical_extract_geometry, flash_extract_geometry

from ..models.autoencoders import TripoSGVAEModel
from ..models.transformers import TripoSGDiTModel
from .pipeline_triposg_output import TripoSGPipelineOutput
from .pipeline_utils import TransformerDiffusionMixin

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class TripoSGPipeline(DiffusionPipeline, TransformerDiffusionMixin):
    """
    Pipeline for image-to-3D generation + (اختياري) conditioning نصّي عربي/إنجليزي.
    """

    def __init__(
        self,
        vae: TripoSGVAEModel,
        transformer: TripoSGDiTModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_encoder_dinov2: Dinov2Model,
        feature_extractor_dinov2: BitImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder_dinov2=image_encoder_dinov2,
            feature_extractor_dinov2=feature_extractor_dinov2,
        )

        # =========================
        # Arabic-Aware Conditioning
        # =========================
        # نحضّر مُكيّف النص — لا يوقف البايبلاين لو فشل (يصبح اختياري)
        self.cond = None
        try:
            # نلفّ الـ CLIP الإنجليزي في واجهة بسيطة تُعيد [B, D]
            from torch import nn

            class EnglishCLIPWrapper(nn.Module):
                def __init__(self, clip_model: CLIPTextModel, tokenizer: CLIPTokenizer, device: str = "cuda"):
                    super().__init__()
                    self.clip = clip_model
                    self.tok = tokenizer
                    self.device = device

                @torch.no_grad()
                def forward(self, prompts: List[str]) -> torch.Tensor:
                    tokens = self.tok(prompts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    out = self.clip(**tokens)
                    # نستخدم CLS (أو pooler_output إن توفر)
                    if hasattr(out, "pooler_output") and out.pooler_output is not None:
                        emb = out.pooler_output
                    else:
                        emb = out.last_hidden_state[:, 0]
                    return emb

            self.text_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.text_enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
            english_dim = self.text_enc.config.hidden_size

            self.en_wrap = EnglishCLIPWrapper(self.text_enc, self.text_tok, device="cuda")

            self.cond = ArabicAwareConditioner(
                english_encoder=self.en_wrap,
                english_dim=english_dim,
                fusion_mode="gate",       # أو "attn"
                out_dim=english_dim,
                device="cuda",
                dtype=torch.float32,
            )
            print("✅ Arabic-Aware Conditioning: ready")
        except Exception as e:
            print("⚠️ Arabic-Aware Conditioning disabled (setup error):", e)
            self.cond = None

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def decode_progressive(self):
        return self._decode_progressive

    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder_dinov2.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor_dinov2(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder_dinov2(image).last_hidden_state          # [B, S, D]
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0) # [B*K, S, D]
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds

    def prepare_latents(
        self,
        batch_size,
        num_tokens,
        num_channels_latents,
        dtype,
        device,
        generator,
        latents: Optional[torch.Tensor] = None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (batch_size, num_tokens, num_channels_latents)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Optional[str] = None,                # ← جديد: برومبت عربي/إنجليزي اختياري
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_shapes_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        bounds: Union[Tuple[float], List[float], float] = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 9,
        flash_octree_depth: int = 9,
        use_flash_decoder: bool = True,
        return_dict: bool = True,
    ):
        # 1) إعداد باراميترات النداء
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs

        # 2) حساب حجم الدفعة
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = self._execution_device

        # 3) ترميز الشرط (الصورة)
        image_embeds, negative_image_embeds = self.encode_image(
            image, device, num_shapes_per_prompt
        )  # image_embeds: [B*K, S, D]

        # 3.1) (اختياري) إضافة تمثيل نصّي عربي/إنجليزي ودمجه مع تمثيل الصورة
        if self.cond is not None and prompt is not None and isinstance(prompt, str) and len(prompt.strip()) > 0:
            try:
                # نكوّن قائمة برومبتات بطول الدُفعة (نفس النص للجميع حالياً)
                prompts = [prompt] * batch_size
                text_emb = self.cond(prompts)  # [B, D_txt]
                # نكرر مثل تكرار الصورة (num_shapes_per_prompt), ثم نوسع عبر طول التسلسل S
                text_emb = text_emb.repeat_interleave(num_shapes_per_prompt, dim=0)        # [B*K, D]
                text_emb = text_emb.unsqueeze(1).expand(-1, image_embeds.size(1), -1)     # [B*K, S, D]
                # المزج: ببساطة جمع (يمكن لاحقاً Attention أدق)
                image_embeds = image_embeds + text_emb
                print("🔠 Applied Arabic/English text conditioning.")
            except Exception as e:
                print("⚠️ Text conditioning skipped due to error:", e)

        # 3.2) CFG
        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)

        # 4) تحضير الجدول الزمني
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5) تحضير اللاتنت
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_shapes_per_prompt,
            num_tokens,
            num_channels_latents,
            image_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6) حلقة إزالة الضجيج
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_image - noise_pred_uncond
                    )

                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        # 7) استخراج الشبكة (المش)
        if not use_flash_decoder:
            geometric_func = lambda x: self.vae.decode(latents, sampled_points=x).sample
            output = hierarchical_extract_geometry(
                geometric_func,
                device,
                bounds=bounds,
                dense_octree_depth=dense_octree_depth,
                hierarchical_octree_depth=hierarchical_octree_depth,
            )
        else:
            self.vae.set_flash_decoder()
            output = flash_extract_geometry(
                latents,
                self.vae,
                bounds=bounds,
                octree_depth=flash_octree_depth,
            )
        meshes = [trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1]) for mesh_v_f in output]

        # تفريغ النماذج من الذاكرة عند الحاجة
        self.maybe_free_model_hooks()

        if not return_dict:
            return (output, meshes)

        return TripoSGPipelineOutput(samples=output, meshes=meshes)
