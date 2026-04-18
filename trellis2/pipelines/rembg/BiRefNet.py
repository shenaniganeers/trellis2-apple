import os
from pathlib import Path
from typing import *
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image


DEFAULT_RMBG_REPO = "briaai/RMBG-2.0"
RMBG_FALLBACK_REPOS = (
    "athena2634/RMBG-2.0",
)


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        self.model_name = model_name
        self.model = None
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self._device = 'cpu'

    def _load_model(self):
        override_model = os.environ.get("TRELLIS2_RMBG_MODEL")
        candidates = []
        for candidate in [override_model, self.model_name, *RMBG_FALLBACK_REPOS]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        last_error = None
        for candidate in candidates:
            try:
                source = Path(candidate) if candidate else None
                model = AutoModelForImageSegmentation.from_pretrained(
                    str(source) if source and source.exists() else candidate,
                    trust_remote_code=True,
                )
                model.eval()
                model.to(self._device)
                if candidate != self.model_name:
                    print(f"[RMBG] Using background remover weights from {candidate}")
                self.model = model
                return
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            "Unable to load background remover weights. Set HF_TOKEN after being granted "
            f"access to {self.model_name}, or point TRELLIS2_RMBG_MODEL at an accessible "
            "Hugging Face repo or local directory."
        ) from last_error

    def _ensure_model(self):
        if self.model is None:
            self._load_model()

    def to(self, device):
        self._device = device
        if self.model is not None:
            self.model.to(device)

    def cuda(self):
        self.to('cuda')

    def cpu(self):
        self.to('cpu')

    def __call__(self, image: Image.Image) -> Image.Image:
        self._ensure_model()
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    
