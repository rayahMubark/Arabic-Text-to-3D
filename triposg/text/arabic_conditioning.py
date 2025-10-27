import torch
import torch.nn as nn
from typing import List, Optional
from langdetect import detect

# ملاحظة: سنستخدم sentence-transformers كـ Arabic encoder
from sentence_transformers import SentenceTransformer


class ConditionFusion(nn.Module):
    """
    دمج تعلّمي بسيط:
    - 'gate': gate scalar/vector يتعلّم أوزان المزج بين العربي/الإنجليزي
    - 'attn': attention خفيف بين (E_ar, E_en) ثم إسقاط
    """
    def __init__(self, dim_in: int, dim_out: int, mode: str = "gate"):
        super().__init__()
        self.mode = mode
        if mode == "gate":
            self.gate = nn.Sequential(
                nn.Linear(dim_in * 2, dim_in),
                nn.GELU(),
                nn.Linear(dim_in, 1),
                nn.Sigmoid(),
            )
        elif mode == "attn":
            self.q_proj = nn.Linear(dim_in, dim_in)
            self.k_proj = nn.Linear(dim_in, dim_in)
            self.v_proj = nn.Linear(dim_in, dim_in)
            self.out_proj = nn.Linear(dim_in, dim_out)
        else:
            raise ValueError("mode must be 'gate' or 'attn'")

        if mode == "gate":
            self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, emb_ar: torch.Tensor, emb_en: torch.Tensor) -> torch.Tensor:
        """
        emb_*: [B, D]
        return: [B, dim_out]
        """
        if self.mode == "gate":
            x = torch.cat([emb_ar, emb_en], dim=-1)      # [B, 2D]
            alpha = self.gate(x)                          # [B, 1] بين 0..1
            fused = alpha * emb_en + (1 - alpha) * emb_ar # مزج ديناميكي
            return self.proj(fused)                       # إسقاط لأبعاد الباكبون
        else:  # attn
            # نعتبر توكن واحد لكل تمثيل (جملة)، نوسع لبعد التتابع = 1
            q = self.q_proj(emb_en).unsqueeze(1)  # [B, 1, D]
            k = self.k_proj(emb_ar).unsqueeze(1)  # [B, 1, D]
            v = self.v_proj(emb_ar).unsqueeze(1)  # [B, 1, D]
            attn = torch.softmax((q @ k.transpose(1, 2)) / (emb_en.size(-1) ** 0.5), dim=-1)  # [B,1,1]
            fused = (attn @ v).squeeze(1)  # [B, D]
            return self.out_proj(fused)    # [B, dim_out]


class ArabicAwareConditioner(nn.Module):
    """
    مُكيّف نص عربي/إنجليزي:
      - english_encoder: دالة ترجع embedding إنجليزي (مثلاً CLIPTextModel output)
      - english_dim: بعد التمثيل الإنجليزي
      - arabic_model_name: موديل جمل للعربي/متعدد اللغات
      - fusion_mode: 'gate' (افتراضي) أو 'attn'
      - out_dim: البُعد الذي يتوقعه الباكبون (عادة نفس english_dim)
    """
    def __init__(
        self,
        english_encoder: nn.Module,
        english_dim: int,
        arabic_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        fusion_mode: str = "gate",
        out_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.english_encoder = english_encoder
        self.english_dim = english_dim
        self.out_dim = out_dim or english_dim
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float32

        # Arabic encoder
        self.ar_model = SentenceTransformer(arabic_model_name, device=str(device) if device else None)

        # Fusion
        self.fusion = ConditionFusion(dim_in=english_dim, dim_out=self.out_dim, mode=fusion_mode)

    @torch.no_grad()
    def encode_en(self, prompts: List[str]) -> torch.Tensor:
        """
        يُفترض english_encoder يُعيد [B, D] (مثلاً أخذ CLS/pooled output من CLIP)
        هنا نحافظ على واجهة بسيطة: english_encoder(prompts) -> [B,D]
        """
        return self.english_encoder(prompts)  # لازم تكوني مضبطة الـ encoder الخارجي

    @torch.no_grad()
    def encode_ar(self, prompts: List[str]) -> torch.Tensor:
        # sentence-transformers ترجع numpy، نُحوّله Tensor ونسنده على الجهاز
        emb = self.ar_model.encode(prompts, convert_to_numpy=True, normalize_embeddings=True)
        emb = torch.from_numpy(emb).to(self.device, self.dtype)  # [B, D_ar]
        # نسقطه لنفس بُعد الإنجليزي لو اختلف (MiniLM يعطي 384 عادة)
        if emb.size(-1) != self.english_dim:
            proj = getattr(self, "_ar_proj", None)
            if proj is None:
                self._ar_proj = nn.Linear(emb.size(-1), self.english_dim, bias=False).to(self.device)
                nn.init.xavier_uniform_(self._ar_proj.weight)
                proj = self._ar_proj
            emb = proj(emb)
        return emb  # [B, D_en]

    @torch.no_grad()
    def forward(self, prompts: List[str]) -> torch.Tensor:
        """
        يعود بتضمين مُدمج [B, out_dim]
        - يُحاول كشف اللغة تلقائياً لكل برومبت
        - لو إنجليزي فقط: يرجع إنجليزي
        - لو عربي فقط: يرجع عربي مُسقَط
        - لو خليط: يدمج حسب fusion
        """
        # اكتشاف اللغة (سريع وبسيط)
        langs = []
        for p in prompts:
            try:
                langs.append(detect(p))
            except Exception:
                langs.append("en")

        has_ar = any(l.startswith("ar") for l in langs)
        has_en = any(not l.startswith("ar") for l in langs)

        # نُشفّر حسب الحاجة
        E_en = self.encode_en(prompts) if has_en else None    # [B, D]
        E_ar = self.encode_ar(prompts) if has_ar else None    # [B, D] (بعد الإسقاط إن لزم)

        if has_ar and has_en:
            fused = self.fusion(E_ar, E_en)                   # [B, out_dim]
            return fused
        elif has_ar:
            # عربي فقط → إسقاط إلى out_dim
            if self.out_dim != E_ar.size(-1):
                out_proj = getattr(self, "_out_proj_ar", None)
                if out_proj is None:
                    self._out_proj_ar = nn.Linear(E_ar.size(-1), self.out_dim, bias=False).to(self.device)
                    nn.init.xavier_uniform_(self._out_proj_ar.weight)
                    out_proj = self._out_proj_ar
                return out_proj(E_ar)
            return E_ar
        else:
            # إنجليزي فقط → إسقاط/مطابقة out_dim (غالباً مساوي)
            if self.out_dim != E_en.size(-1):
                out_proj = getattr(self, "_out_proj_en", None)
                if out_proj is None:
                    self._out_proj_en = nn.Linear(E_en.size(-1), self.out_dim, bias=False).to(self.device)
                    nn.init.xavier_uniform_(self._out_proj_en.weight)
                    out_proj = self._out_proj_en
                return out_proj(E_en)
            return E_en
