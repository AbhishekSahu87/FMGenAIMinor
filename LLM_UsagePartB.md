# LLM Usage Report

## Model

We used **Gemini AI (Google, September 2025 release)** for all experiments. The model supports long-context reasoning and multiple languages, making it suitable for testing robustness and context retention.

## Decoding Settings

* **Temperature:** 0.2
* **Top-p:** 1.0
* **Max tokens:** 64 (normal QA), 32 (needle retrieval)
* **Repetitions:** 2 per condition to reduce stochastic variance

## Languages Tested

* **L1:** English (high-resource)
* **L2:** Hindi (medium-resource)
* **L3:** Hinglish (code-mixed dialect)
* **CS:** Code-Switch (different language for Q/A)

## Tasks

1. **Multilingual QA Robustness**

   * Tested model on 20 items across L1, L2, L3, and CS conditions.
   * Metrics: Accuracy (0/1), Fluency (1–5 human rating).
   * Mitigation: Language pinning, 3-shot bilingual exemplars.

2. **Needle-in-a-Haystack & Lost-in-the-Middle**

   * Constructed ~10k token context with unique inserted fact.
   * Needle positions: **Start, Middle, End**.
   * Query: *"What is the secret ticket code? Final: <code>."*
   * Logged total tokens, needle index, prediction, and correctness.
   * Mitigation: **Chunk + Retrieve**, which improved middle retrieval from 50% → 100%.

## Outputs

* **Tables:** Per-condition accuracy & fluency with 95% CIs.
* **Plots:** Accuracy drop at middle position (Lost-in-the-Middle).
* **Failure Cases:** Documented with short diagnosis.
* **Mitigation Results:** Before/after delta tables included.
* **CSV Files:** Contain all prompts, predictions, correctness, fluency ratings, and metadata (token counts, positions).

---

This document serves as the usage and methodology record for all LLM experiments conducted with Gemini AI.
