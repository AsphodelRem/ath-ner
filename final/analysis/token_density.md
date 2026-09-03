# Latin vs Cyrillic tokenizer density

All tokenizers were loaded from local directories with `local_files_only=True`. Tokenizer counts exclude model special tokens.

Reproduce from the repository root with `.venv/bin/python scripts/analyze_token_density.py`.

A pure-Latin document contains Latin letters and no Cyrillic or other-script letters; pure Cyrillic is defined symmetrically. Unicode modifier letters (including common Uzbek apostrophe characters) are script-neutral. Letters are Unicode Latin/Cyrillic letters. A word is a whitespace-delimited field containing at least one Unicode letter or digit.

Corpus rates are micro-averages (`total tokens / total denominator`). Median rates are medians of per-document rates. Ratios are Cyrillic / Latin and gaps are Cyrillic - Latin.

## Dataset selection

| Split | All | Pure Latin | Pure Cyrillic | Mixed Latin/Cyrillic | Other/multiscript | Letterless |
|---|---:|---:|---:|---:|---:|---:|
| train | 13,000 | 7,749 | 2,195 | 2,927 | 129 | 0 |
| dev | 1,500 | 886 | 271 | 332 | 11 | 0 |
| combined | 14,500 | 8,635 | 2,466 | 3,259 | 140 | 0 |

## Combined train + dev

| Tokenizer | Script | Docs | Tokens | Tokens/100 letters | Tokens/word | Median tokens/doc | Median tokens/100 letters | Median tokens/word |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mmbert_base | latin | 8,635 | 1,575,665 | 44.682 | 2.898 | 46.0 | 42.878 | 2.800 |
| mmbert_base | cyrillic | 2,466 | 338,836 | 56.279 | 3.541 | 54.0 | 54.714 | 3.364 |
| xlm_roberta_base | latin | 8,635 | 1,218,184 | 34.545 | 2.241 | 39.0 | 35.714 | 2.241 |
| xlm_roberta_base | cyrillic | 2,466 | 296,505 | 49.248 | 3.099 | 47.0 | 48.320 | 2.984 |
| baseline_distilmbert | latin | 8,635 | 1,501,721 | 42.585 | 2.762 | 46.0 | 42.105 | 2.729 |
| baseline_distilmbert | cyrillic | 2,466 | 336,082 | 55.821 | 3.513 | 59.0 | 53.444 | 3.333 |

## Cyrillic overhead on combined train + dev

| Tokenizer | Corpus T/100 ratio | Corpus T/100 gap | Corpus T/word ratio | Corpus T/word gap | Median T/100 ratio | Median T/100 gap | Median T/word ratio | Median T/word gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mmbert_base | 1.260 | +11.597 | 1.222 | +0.643 | 1.276 | +11.837 | 1.201 | +0.564 |
| xlm_roberta_base | 1.426 | +14.703 | 1.383 | +0.858 | 1.353 | +12.606 | 1.331 | +0.743 |
| baseline_distilmbert | 1.311 | +13.236 | 1.272 | +0.751 | 1.269 | +11.338 | 1.221 | +0.604 |

## Train corpus rates

| Tokenizer | Latin tokens/100 letters | Cyrillic tokens/100 letters | Cyr/Lat ratio | Latin tokens/word | Cyrillic tokens/word | Cyr/Lat ratio |
|---|---:|---:|---:|---:|---:|---:|
| mmbert_base | 44.723 | 56.188 | 1.256 | 2.896 | 3.539 | 1.222 |
| xlm_roberta_base | 34.565 | 49.137 | 1.422 | 2.239 | 3.095 | 1.383 |
| baseline_distilmbert | 42.603 | 55.763 | 1.309 | 2.759 | 3.512 | 1.273 |

## Dev corpus rates

| Tokenizer | Latin tokens/100 letters | Cyrillic tokens/100 letters | Cyr/Lat ratio | Latin tokens/word | Cyrillic tokens/word | Cyr/Lat ratio |
|---|---:|---:|---:|---:|---:|---:|
| mmbert_base | 44.318 | 56.949 | 1.285 | 2.913 | 3.559 | 1.221 |
| xlm_roberta_base | 34.363 | 50.070 | 1.457 | 2.259 | 3.129 | 1.385 |
| baseline_distilmbert | 42.426 | 56.253 | 1.326 | 2.789 | 3.515 | 1.260 |

## Local tokenizer paths

- `mmbert_base`: `artifacts/pretrained/mmbert-base` (TokenizersBackend)
- `xlm_roberta_base`: `artifacts/experiments/xlm-roberta-base/model` (XLMRobertaTokenizer)
- `baseline_distilmbert`: `artifacts/baseline/model` (BertTokenizer)
