# scripts/briarmbg.py
import torch
from transformers import AutoModel

class BriaRMBG:
    @staticmethod
    def from_pretrained(path_or_repo: str):
        model = AutoModel.from_pretrained(path_or_repo)
        return model.to(torch.float16)
