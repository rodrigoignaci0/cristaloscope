# Cristaloscope
### LLM Internal Space Visualizer & Hallucination Interceptor

Cristaloscope is a research toolkit for visualizing and intervening in the internal geometry of large language models. Built on NE-OS Research findings, it exposes three phenomena: the Three Phases of internal representation, pre-generation hallucination detection from hidden states (AUC 0.885), and surgical activation steering that corrects wrong answers without modifying model weights.

---

## Key Findings

- **Three Phases**: every LLM exhibits chaos (L0–8), semantic organization (L9–23), and crystallization (L24+) — a universal internal structure across architectures.
- **Pre-generation hallucination detection** at AUC 0.885 from hidden states at layer 12, before any token is generated.
- Only **100 of 4096 dimensions** carry the uncertainty signal — 41x compression with no information loss.
- **Activation steering** changes wrong answers to correct ones without modifying any weights.
- **Interceptor** catches 75% of hallucinations on unseen prompts (recall), trained on just 435 examples.

---

## Install

```bash
pip install git+https://github.com/rodrigoignaci0/cristaloscope
```

---

## Quick Start

### CristalAnalyzer — extract and visualize hidden states

```python
from cristaloscope import CristalAnalyzer

analyzer = CristalAnalyzer(model_name="Qwen/Qwen2.5-7B-Instruct")
states = analyzer.extract("The capital of France is")

analyzer.plot_pca_layers(states)          # layer-by-layer PCA trajectory
analyzer.plot_cosine_heatmap(states)      # inter-layer similarity matrix
```

### ThreePhases — identify phase boundaries

```python
from cristaloscope import ThreePhases

phases = ThreePhases(model_name="Qwen/Qwen2.5-7B-Instruct")
result = phases.detect("Explain quantum entanglement")

print(result.boundaries)   # e.g. {'chaos': (0, 8), 'semantic': (9, 23), 'crystal': (24, 31)}
print(result.jump_layer)   # layer where crystallization spike occurs
phases.plot(result)
```

### HallucinationInterceptor — detect hallucinations before generation

```python
from cristaloscope import HallucinationInterceptor

interceptor = HallucinationInterceptor(model_name="Qwen/Qwen2.5-7B-Instruct", probe_layer=12)

# fit expects list of prompts and binary labels (1 = hallucination)
interceptor.fit(prompts_train, labels_train)

predictions = interceptor.predict(prompts_test)   # returns probabilities
print(f"AUC: {interceptor.auc_score(prompts_test, labels_test):.3f}")
```

### ActivationSteering — correct outputs without touching weights

```python
from cristaloscope import ActivationSteering

steerer = ActivationSteering(model_name="Qwen/Qwen2.5-7B-Instruct")

# fit learns a steering vector from correct/incorrect response pairs
steerer.fit(wrong_prompts, correct_prompts, layer=20)

output = steerer.steer("Who invented the telephone?", strength=1.5)
print(output.corrected_text)
```

---

## Research

Cristaloscope implements findings from NE-OS Research on LLM internal geometry. The Three Phases phenomenon, hallucination probe, and steering methodology were validated on:

- Qwen2.5-7B-Instruct
- Mistral-7B-v0.1
- Phi-2

The crystallization jump (abrupt emergence of the correct token at a single layer, rather than gradual convergence) was empirically confirmed across all three architectures, with cosine similarity projections > 0.99 between model families — consistent with the Platonic Representation Hypothesis.

Full methodology and experimental results available at [NE-OS Research](https://github.com/rodrigoignaci0).

---

## Citation

```bibtex
@software{cristaloscope2026,
  author = {NE-OS Research},
  title  = {Cristaloscope: LLM Internal Space Visualizer},
  year   = {2026},
  url    = {https://github.com/rodrigoignaci0/cristaloscope}
}
```
