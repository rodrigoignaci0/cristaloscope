"""
CristalAnalyzer — extrae y analiza el espacio interno de cualquier LLM HuggingFace.
"""
import sys
sys.modules.setdefault('torchaudio', None)

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class LayerProfile:
    layer: int
    logitlens_rank: int          # rank del token correcto vía logit lens
    logitlens_prob: float        # probabilidad del token correcto
    cosine_to_next: float        # cosine similarity con capa siguiente
    delta_norm: float            # ||h_{l+1} - h_l|| / ||h_l||
    top1_token: str              # token con mayor logit en esta capa
    top1_prob: float


@dataclass
class CristalResult:
    model_id: str
    prompt: str
    answer: str
    n_layers: int
    crystal_layer: Optional[int]   # primera capa donde rank < threshold
    crystal_type: str              # "abrupt" | "gradual" | "never"
    phases: dict                   # {layer: "chaos"|"semantic"|"crystallization"}
    profiles: list[LayerProfile]
    hidden_states: Optional[np.ndarray] = field(default=None, repr=False)  # (n_layers, dim)


class CristalAnalyzer:
    def __init__(self, model_id: str, device: str = "auto", rank_threshold: int = 100):
        self.model_id = model_id
        self.rank_threshold = rank_threshold
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16,
            device_map=self.device, trust_remote_code=True
        )
        self.model.eval()
        self.n_layers = self.model.config.num_hidden_layers

        # Obtener lm_head y norm final
        self._lm_head = self.model.lm_head
        self._norm = getattr(self.model.model, "norm", None)

    def _logitlens_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Proyecta hidden state h al vocabulario vía logit lens."""
        h = h.float()
        if self._norm is not None:
            h = self._norm(h.unsqueeze(0)).squeeze(0)
        return self._lm_head(h.to(self._lm_head.weight.dtype)).float()

    def analyze(self, prompt: str, answer: str, store_hidden: bool = False) -> CristalResult:
        ans_ids = self.tok(answer, add_special_tokens=False).input_ids
        ans_id  = ans_ids[0]

        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model(ids, output_hidden_states=True)

        hidden_states = out.hidden_states  # tuple: (n_layers+1,) cada (1, seq, dim)

        profiles = []
        crystal_layer = None
        prev_h = None

        for l, hs in enumerate(hidden_states[1:]):  # skip embedding layer
            h = hs[0, -1]  # último token

            # Logit lens
            logits = self._logitlens_logits(h)
            probs  = torch.softmax(logits, dim=-1)
            rank   = int((logits > logits[ans_id]).sum().item())
            prob   = float(probs[ans_id].item())
            top1_id = int(logits.argmax().item())
            top1_tok = self.tok.decode([top1_id]).strip()
            top1_prob = float(probs[top1_id].item())

            # Geometría
            h_cpu = h.float().cpu()
            if prev_h is not None:
                cos = float(torch.nn.functional.cosine_similarity(
                    prev_h.unsqueeze(0), h_cpu.unsqueeze(0)).item())
                delta = float((h_cpu - prev_h).norm() / (prev_h.norm() + 1e-8))
            else:
                cos, delta = 1.0, 0.0
            prev_h = h_cpu

            profiles.append(LayerProfile(
                layer=l, logitlens_rank=rank, logitlens_prob=prob,
                cosine_to_next=cos, delta_norm=delta,
                top1_token=top1_tok, top1_prob=top1_prob
            ))

            if crystal_layer is None and rank < self.rank_threshold:
                crystal_layer = l

        # Clasificar tipo de cristalización
        if crystal_layer is None:
            crystal_type = "never"
        else:
            # Abrupta: pasa de rank>1000 a rank<100 en ≤2 capas
            if crystal_layer > 0 and profiles[crystal_layer - 1].logitlens_rank > 1000:
                crystal_type = "abrupt"
            else:
                crystal_type = "gradual"

        # Fases por capa
        p1 = int(self.n_layers * 0.30)
        p2 = int(self.n_layers * 0.80)
        phases = {}
        for p in profiles:
            if p.layer < p1:
                phases[p.layer] = "chaos"
            elif p.layer < p2:
                phases[p.layer] = "semantic"
            else:
                phases[p.layer] = "crystallization"

        hidden_np = None
        if store_hidden:
            hidden_np = np.stack([
                hs[0, -1].float().cpu().numpy() for hs in hidden_states[1:]
            ])

        return CristalResult(
            model_id=self.model_id,
            prompt=prompt,
            answer=answer,
            n_layers=self.n_layers,
            crystal_layer=crystal_layer,
            crystal_type=crystal_type,
            phases=phases,
            profiles=profiles,
            hidden_states=hidden_np,
        )

    def analyze_batch(self, prompts: list[tuple[str, str]], store_hidden: bool = False):
        return [self.analyze(p, a, store_hidden) for p, a in prompts]
