# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Amir Mohammadi  <amir.mohammadi@idiap.ch>
# SPDX-FileContributor: Samuel Neugber  <samuel.neugber@idiap.ch>
#
# SPDX-License-Identifier: MIT

from collections import namedtuple
from pathlib import Path
import numpy as np
import torch as pt
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from  Cue_Net.model.CueNet import CueNet

def preprocess_image(img: np.ndarray) -> pt.Tensor:
    transforms_list = [transforms.ToPILImage(), transforms.ToTensor()]
    transforms_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    transform = transforms.Compose(transforms_list)
    img = transform(img)
    return img

class CueNetPipeline:
    def __init__(
        self,
        model_path: str = Path(__file__).parent / "Cue_Net/checkpoint/ICCV2025_DeepID_Localization_1st_Sunlight.pt", 
        device: str = "",
    ):
        self.model_path = model_path
        self.device = device or "cuda" if pt.cuda.is_available() else "cpu"
        print(f"Model inference will run on {self.device}")
        self._model = None

    @property
    def model(self) -> nn.Module:
        """Load the model."""
        if self._model is None:
            model = CueNet(backbone='backbone', num_classes=1, pretrained=True).to(self.device)
            model_param = torch.load(self.model_path, map_location='cpu')
            model.load_state_dict(model_param['model_state_dict'])
            del model_param
            self._model = model.eval()
        return self._model

    def _forward(self, batch: pt.Tensor) -> tuple[pt.Tensor, ...]:
        """Run forward: -> mask_pred, conf, det, npp"""
        with pt.inference_mode():
            batch = batch.to(device=self.device)
            device_data = pt.as_tensor(batch, device=self.device)
            return self.model(device_data)
        
    def detect(self, img: pt.Tensor) -> float:
        """Run prediction."""
        batch = img[None, ...]
        prob, _, _,  _ = self._forward(batch=batch)
        return self._compute_score(prob)

    def _compute_score(self, prob):
        prob = prob.detach().cpu().numpy()
        score = np.max(prob)
        # Model outputs 0 for pristine pixels and 1 for forged pixels
        return float(1.0- score)

    def localize(self, img: pt.Tensor) -> np.ndarray:
        """Run prediction."""
        batch = img[None, ...]
        pred, _, _, _ = self._forward(batch=batch)
        return self._compute_mask(pred)

    def _compute_mask(self, prob):
        prob = prob.detach().cpu().numpy()
        pred = (prob > 0.5).squeeze(0).squeeze(0)  
        pred = ~pred
        return pred.astype('bool')  
    
    def detect_and_localize(self, img: pt.Tensor) -> tuple[float, np.ndarray]:
        """Run detection and localization in one forward pass."""
        batch = img[None, ...]
        pred, _,_, _ = self._forward(batch=batch)
        score = self._compute_score(pred)
        mask = self._compute_mask(pred)
        return score, mask
