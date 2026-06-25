"""
Ejemplo de uso — Cristaloscope
"""
from cristaloscope import CristalAnalyzer, CristalVisualizer
import matplotlib.pyplot as plt

# 1. Analizar un prompt en un modelo
analyzer = CristalAnalyzer("Qwen/Qwen2.5-7B-Instruct")

result = analyzer.analyze(
    prompt="The capital of France is",
    answer="Paris",
    store_hidden=True   # necesario para trayectoria 2D
)

print(f"Crystal layer: L{result.crystal_layer} ({result.crystal_type})")
print(f"Final rank: {result.profiles[-1].logitlens_rank}")

# 2. Reporte completo
fig = CristalVisualizer.full_report(result)
fig.savefig("report_qwen_paris.png", dpi=150, bbox_inches="tight")
print("Guardado: report_qwen_paris.png")

# 3. Comparar tres modelos en el mismo prompt
models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-v0.1",
    "google/gemma-2-2b",
]

results = []
for model_id in models:
    a = CristalAnalyzer(model_id)
    r = a.analyze("The capital of France is", "Paris")
    results.append(r)
    print(f"{model_id}: crystal=L{r.crystal_layer}")

fig2 = CristalVisualizer.compare_models(results)
fig2.savefig("compare_models.png", dpi=150, bbox_inches="tight")
print("Guardado: compare_models.png")
