"""Run a small reproducible Cristaloscope analysis on Qwen2.5-0.5B.

This example is intended as a CPU-friendly smoke test. It downloads the model
from Hugging Face on first run, analyzes one prompt, and saves a visual report.
"""
import sys
from pathlib import Path

sys.modules.setdefault("torchaudio", None)

import matplotlib

matplotlib.use("Agg")

from cristaloscope import CristalAnalyzer, CristalVisualizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The capital of France is"
ANSWER = "Paris"
OUTPUT_PATH = Path(__file__).with_name("qwen_0_5b_report.png")


def main() -> None:
    print(f"Loading {MODEL_ID}...")
    analyzer = CristalAnalyzer(MODEL_ID, device="cpu")

    result = analyzer.analyze(
        prompt=PROMPT,
        answer=ANSWER,
        store_hidden=True,
    )

    print(f"Crystal layer: L{result.crystal_layer} ({result.crystal_type})")
    print(f"N layers: {result.n_layers}")
    print(f"Final rank: {result.profiles[-1].logitlens_rank}")
    print(f"Top-1 last layer: {result.profiles[-1].top1_token!r}")

    fig = CristalVisualizer.full_report(result)
    fig.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight", facecolor="#0d0d1a")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
