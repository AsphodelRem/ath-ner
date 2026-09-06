---
license: apache-2.0
language:
  - uz
  - ru
  - en
base_model: google/rembert
pipeline_tag: token-classification
tags:
  - ner
  - uzbek
  - exact-span
  - domain-adaptive-pretraining
---

# Uzbek exact-span NER (RemBERT + domain-adaptive MLM)

Named entity recognition for Uzbek text with exact character offsets.
Three classes: `ORG` (organizations and brands), `NAME` (people),
`GEO` (geographic entities).

Handles both Latin and Cyrillic scripts, plus Russian and English fragments.

## Metrics

Exact-span match on a held-out set (1500 documents, 7698 entities). An entity
counts only when the class and both boundaries match exactly.

| class | precision | recall | F1 |
|-------|-----------|--------|------|
| ORG | 0.8872 | 0.8665 | 0.8767 |
| NAME | 0.9182 | 0.8814 | 0.8994 |
| GEO | 0.9153 | 0.9096 | 0.9124 |
| **micro** | **0.9065** | **0.8862** | **0.8962** |

## Intended use

Extracting people, organizations and locations from Uzbek news and social media
text. Exact character offsets make the output suitable for downstream analytics:
entity frequency and co-occurrence statistics, media monitoring, building entity
indexes over document collections, and linking mentions back to their exact
position in the source text.

## Important: weights alone are not enough

Tags are predicted in the **BILOU** scheme and decoded with the **Viterbi
algorithm** using the transition matrix in `transitions.json`. That file is not a
standard Hugging Face artifact, so `AutoModelForTokenClassification.from_pretrained`
will ignore it. Without it decoding falls back to per-token argmax and F1 drops by
roughly one point. Boundary post-processing (quotes, brackets, affixes) is also
applied and also matters for exact-span scoring.

Run the model through the solution repository instead:

```bash
python -m solution.predict \
  --model-dir <directory with these files> \
  --input texts.jsonl \
  --output predictions.jsonl
```

Input is JSON Lines with `hash` and `text`. Output is `hash` plus a list of
`{"label", "start", "end"}`, where offsets are zero-based Unicode character
positions and `end` is exclusive.

Inference settings come from `run_config.json`: window 512 with stride 128, BILOU,
Viterbi in `conditional` mode with transition weight 0.4.

## Training

Two stages.

**Domain adaptation.** `google/rembert` was further pretrained with masked language
modeling on an Uzbek corpus of 405,585 passages (~220M tokens): Telegram blogs and
news from `tahrirchi/uz-crawl`, Cyrillic and Latin books from
`tahrirchi/uz-books-v2`, plus 8% Russian and English Wikipedia to preserve
multilinguality. Whole-word masking at 15%, 16,800 steps, held-out perplexity 11.5.

**Fine-tuning.** Token classification in BILOU, window 512 with stride 128,
effective batch 128, learning rate 3e-5, checkpoint selected by exact-span micro-F1.

Full reproduction protocol is in `REPRODUCE.md` of the solution repository.

## Limitations

Trained on news and blog text; other genres were not evaluated.

## Attribution

The base model `google/rembert` is licensed under Apache 2.0. The adaptation corpus
was assembled from `tahrirchi/uz-crawl`, `tahrirchi/uz-books-v2` and
`wikimedia/wikipedia`; their respective licenses apply to the underlying texts.

The annotated NER dataset used for fine-tuning was provided by Brand Analytics.
