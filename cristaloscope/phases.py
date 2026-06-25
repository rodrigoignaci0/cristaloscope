"""
ThreePhases — analiza las tres fases internas del proceso de cristalización en LLMs.
"""
import torch
import numpy as np


class ThreePhases:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tok = tokenizer
        self._device = next(model.parameters()).device
        self._lm_head = model.lm_head
        self._norm = getattr(model.model, "norm", None)

    def _get_rank(self, hidden_state: torch.Tensor, answer_id: int) -> int:
        h = hidden_state.float()
        if self._norm is not None:
            h = self._norm(h.unsqueeze(0)).squeeze(0)
        logits = self._lm_head(h.to(self._lm_head.weight.dtype)).float()
        return int((logits > logits[answer_id]).sum().item())

    def analyze(self, prompt: str, answer: str) -> dict:
        ans_id = self.tok(answer, add_special_tokens=False).input_ids[0]
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self._device)
        with torch.no_grad():
            out = self.model(ids, output_hidden_states=True)

        hidden_states = out.hidden_states[1:]  # skip embedding
        n = len(hidden_states)
        ranks, top1_tokens = [], []

        for hs in hidden_states:
            h = hs[0, -1]
            ranks.append(self._get_rank(h, ans_id))
            logits = self._lm_head(h.to(self._lm_head.weight.dtype)).float()
            top1_tokens.append(self.tok.decode([int(logits.argmax())]).strip())

        crystal_layer = next((i for i, r in enumerate(ranks) if r < 100), None)
        crystal_type = "never"
        if crystal_layer is not None:
            crystal_type = "abrupt" if crystal_layer > 0 and ranks[crystal_layer - 1] > 1000 else "gradual"

        p1, p2 = int(n * 0.30), int(n * 0.85)
        phases = {
            "chaos": [0, p1 - 1],
            "organization": [p1, p2 - 1],
            "crystallization": [p2, n - 1],
        }

        return {
            "crystal_layer": crystal_layer,
            "crystal_type": crystal_type,
            "phases": phases,
            "ranks": ranks,
            "top1_tokens": top1_tokens,
        }
