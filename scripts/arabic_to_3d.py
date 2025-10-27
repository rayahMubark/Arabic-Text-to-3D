# scripts/arabic_to_3d.py
# -*- coding: utf-8 -*-
import argparse
import os
import tempfile
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from huggingface_hub import snapshot_download
from diffusers import StableDiffusionXLPipeline

from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from .image_process import prepare_image

# -----------------------------
# Arabic-aware translation (بسيطة)
# -----------------------------
GLOSSARY: Dict[str, str] = {
    "دلة نجدية": "Najdi dallah coffee pot",
    "دلة": "dallah coffee pot",
    "نقش سدو": "Sadu pattern",
    "نقش السدو": "Sadu pattern",
    "تراثي": "traditional",
    "سعودي": "Saudi",
    "نجدي": "Najdi",
    "مزخرفة": "ornamented",
    "مزخرف": "ornamented",
    "خشبي": "wooden",
    "نحاسي": "copper",
    "ذهبي": "golden",
    "فضي": "silver",
}

def _apply_glossary(ar_text: str) -> str:
    t = ar_text
    for ar, en in sorted(GLOSSARY.items(), key=lambda x: -len(x[0])):
        t = t.replace(ar, en)
    return t

def translate_ar_to_en(text: str) -> str:
    text = _apply_glossary(text)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        mdl = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)
        inputs = tok(text, return_tensors="pt").to(mdl.device)
        forced_bos_id = tok.lang_code_to_id["eng_Latn"]
        out = mdl.generate(**inputs, forced_bos_token_id=forced_bos_id, max_length=256)
        return tok.batch_decode(out, skip_special_tokens=True)[0]
    except Exception:
        from transformers import pipeline as hf_pipeline
        trans = hf_pipeline("translation", model="Helsinki-NLP/opus-mt-ar-en",
                            device=0 if device == "cuda" else -1)
        return trans(text)[0]["translation_text"]

# -----------------------------
# SDXL prompt
# -----------------------------
NEGATIVE_PROMPT = (
    "lowres, blurry, noisy, deformed, distorted, bad proportions, extra limbs, "
    "text, watermark, logo, cropped, frame, out of frame, low quality, jpeg artifacts, "
    "sketch, cartoon, duplicate objects, multiple items, collage, grid"
)
def build_sdxl_prompts(concept_en: str) -> Tuple[str, str]:
    positive = (
        f"A single {concept_en}, isolated, centered on a plain white background, "
        "no duplicates, no reflections, no borders, no frame, minimal composition. "
        "realistic product photo, studio lighting, 3/4 angle, sharp focus, high detail."
    )
    return positive, NEGATIVE_PROMPT

# -----------------------------
# SDXL generation
# -----------------------------
def sdxl_generate_image(prompt_en: str, negative_prompt: str, steps: int, out_path: str) -> str:
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch_dtype,
        variant=("fp16" if torch_dtype is torch.float16 else None),
        use_safetensors=True,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        try: pipe.enable_xformers_memory_efficient_attention()
        except Exception: pass
    else:
        pipe.enable_sequential_cpu_offload()
    pipe.enable_vae_slicing()

    img = pipe(
        prompt=prompt_en,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=7.5,
        height=768, width=768
    ).images[0]
    img.save(out_path)
    return out_path

# -----------------------------
# TripoSG reconstruction (no texture)
# -----------------------------
@torch.no_grad()
def image_to_3d_with_triposg(
    image_or_path: Union[str, Image.Image],
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    triposg_dir: str,
    device: str = "cuda",
) -> trimesh.Trimesh:

    pipe: TripoSGPipeline = TripoSGPipeline.from_pretrained(triposg_dir).to(
        device, torch.float16 if torch.cuda.is_available() else torch.float32
    )

    if isinstance(image_or_path, Image.Image):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image_or_path.save(tmp.name)
        path_for_pre = tmp.name
    else:
        path_for_pre = image_or_path

    img_pil = prepare_image(path_for_pre, bg_color=np.array([1.0, 1.0, 1.0]))

    outputs = pipe(
        image=img_pil,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).samples[0]

    verts = outputs[0].astype(np.float32)
    faces = np.ascontiguousarray(outputs[1])
    return trimesh.Trimesh(verts, faces)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="output.glb")
    parser.add_argument("--image-path", type=str, default="generated.png")
    parser.add_argument("--steps-2d", type=int, default=30)
    parser.add_argument("--steps-3d", type=int, default=50)
    parser.add_argument("--guidance-3d", type=float, default=7.0)
    args = parser.parse_args()

    # weights (سيتم تنزيلها إن لم تكن موجودة)
    triposg_dir = "pretrained_weights/TripoSG"
    snapshot_download("VAST-AI/TripoSG", local_dir=triposg_dir, local_dir_use_symlinks=False)

    # 1) ترجمة + بناء برومبت
    en = translate_ar_to_en(args.prompt)
    pos, neg = build_sdxl_prompts(en)
    print(f"[info] EN concept: {en}")

    # 2) توليد صورة 2D
    img_path = sdxl_generate_image(pos, neg, steps=args.steps_2d, out_path=args.image_path)
    print(f"[done] 2D image: {img_path}")

    # 3) تحويل إلى 3D
    img_pil = Image.open(img_path).convert("RGB")
    mesh = image_to_3d_with_triposg(
        image_or_path=img_pil,
        seed=42,
        num_inference_steps=args.steps_3d,
        guidance_scale=args.guidance_3d,
        triposg_dir=triposg_dir,
    )

    mesh.export(args.output_path)
    print(f"[done] 3D mesh saved to: {args.output_path}")

if __name__ == "__main__":
    main()
