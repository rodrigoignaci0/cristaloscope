"""
HallucinationInterceptor — detecta riesgo de alucinación vía hidden states.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


class HallucinationInterceptor:
    def __init__(self, model, tokenizer, layer: int = 12):
        self.model = model
        self.tok = tokenizer
        self.layer = layer
        self.clf = None
        self._device = next(model.parameters()).device

    def _extract_hidden(self, prompt: str) -> np.ndarray:
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self._device)
        with torch.no_grad():
            out = self.model(ids, output_hidden_states=True)
        h = out.hidden_states[self.layer + 1][0, -1]
        return h.float().cpu().numpy()

    def fit(self, prompts: list[str], labels: list[bool]) -> None:
        X = np.stack([self._extract_hidden(p) for p in prompts])
        y = np.array(labels, dtype=int)
        self.clf = LogisticRegression(C=0.1, max_iter=1000)
        self.clf.fit(X, y)

    def predict(self, prompt: str) -> dict:
        if self.clf is None:
            raise RuntimeError("Call fit() before predict()")
        x = self._extract_hidden(prompt).reshape(1, -1)
        score = float(self.clf.predict_proba(x)[0, 1])
        return {
            "score": score,
            "risk": "HIGH_RISK" if score >= 0.4 else "LOW_RISK",
            "threshold": 0.4,
        }
