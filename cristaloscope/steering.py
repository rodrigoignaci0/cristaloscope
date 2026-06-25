"""
ActivationSteering — dirige el comportamiento del modelo via activation patching.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


class ActivationSteering:
    def __init__(self, model, tokenizer, layer: int = 12):
        self.model = model
        self.tok = tokenizer
        self.layer = layer
        self._device = next(model.parameters()).device
        self._uncertainty_vector: np.ndarray | None = None

    def _extract_hidden(self, prompt: str) -> np.ndarray:
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self._device)
        with torch.no_grad():
            out = self.model(ids, output_hidden_states=True)
        return out.hidden_states[self.layer + 1][0, -1].float().cpu().numpy()

    def fit(self, prompts: list[str], labels: list[bool]) -> None:
        X = np.stack([self._extract_hidden(p) for p in prompts])
        y = np.array(labels, dtype=int)
        clf = LogisticRegression(C=0.1, max_iter=1000).fit(X, y)
        vec = clf.coef_[0]
        self._uncertainty_vector = vec / (np.linalg.norm(vec) + 1e-8)

    def _get_uncertainty_vector(self) -> np.ndarray:
        if self._uncertainty_vector is None:
            raise RuntimeError("Call fit() before steer()")
        return self._uncertainty_vector

    def steer(self, prompt: str, alpha: float = 2.0) -> str:
        vec = torch.tensor(
            self._get_uncertainty_vector(), dtype=torch.float16, device=self._device
        )
        handle = None

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden[:, -1, :] += alpha * vec
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

        target = self.model.model.layers[self.layer]
        handle = target.register_forward_hook(hook_fn)
        try:
            ids = self.tok(prompt, return_tensors="pt").input_ids.to(self._device)
            with torch.no_grad():
                out = self.model.generate(ids, max_new_tokens=128, do_sample=False)
            text = self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        finally:
            handle.remove()
        return text
