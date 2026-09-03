# ORG/GEO exact-boundary audit

This audit counts a confusion only when a gold ORG/GEO span has an exact-boundary prediction with the opposite label. Document length is measured in Unicode characters.

## Data

| Split | Documents | Entities | ORG | GEO | Documents <=128 chars |
|---|---:|---:|---:|---:|---:|
| train | 13,000 | 66,083 | 23,420 | 21,445 | 6,068 |
| dev | 1,500 | 7,698 | 2,659 | 2,720 | 663 |

## Training-surface evidence

After Unicode NFKC, casefold, whitespace collapse, train contains 137 surfaces observed as both ORG and GEO.

| Dev ORG/GEO surface signal | Count |
|---|---:|
| seen_as_ORG_or_GEO | 3,654 |
| unseen_as_ORG_or_GEO | 1,725 |
| seen_with_both_labels | 428 |
| training_majority_is_opposite | 43 |

## Key findings

- `span_stacker_bilou_dual_decode_oof` reduces exact-boundary ORG/GEO swaps from 64 to 28 (56.25% fewer).
- In documents <=128 characters, swaps fall from 9 (1.64%) to 4 (0.73%). The short-text swap rates are respectively 1.38x and 1.40x their all-length rates.
- Of the first model's 64 swaps, the second model makes 20 exact-correct, retains 25, and has no exact-boundary prediction for 19. It adds 3 new opposite-label outcomes.
- Source consensus flags 8 spans for annotation review; 7 are also assigned the opposite label by both audited outputs. These remain review candidates because contextual ORG/GEO roles can legitimately override surface priors.

## Exact-boundary ORG/GEO confusions

| Model | Gold ORG | ORG->GEO | Rate | Gold GEO | GEO->ORG | Rate | Total swaps | Rate / ORG+GEO gold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mmbert_bilou_viterbi | 2,659 | 31 | 1.17% | 2,720 | 33 | 1.21% | 64 | 1.19% |
| span_stacker_bilou_dual_decode_oof | 2,659 | 16 | 0.60% | 2,720 | 12 | 0.44% | 28 | 0.52% |

## Document-length slices

| Model | Character-length bucket | ORG+GEO gold | ORG->GEO | GEO->ORG | Total swaps | Swap rate | Exact-correct rate |
|---|---|---:|---:|---:|---:|---:|---:|
| mmbert_bilou_viterbi | 0000-0128 | 548 | 6 | 3 | 9 | 1.64% | 81.93% |
| mmbert_bilou_viterbi | 0129-0512 | 1,296 | 11 | 11 | 22 | 1.70% | 81.40% |
| mmbert_bilou_viterbi | 0513-2048 | 2,457 | 11 | 13 | 24 | 0.98% | 88.64% |
| mmbert_bilou_viterbi | 2049+ | 1,078 | 3 | 6 | 9 | 0.83% | 86.55% |
| span_stacker_bilou_dual_decode_oof | 0000-0128 | 548 | 2 | 2 | 4 | 0.73% | 82.48% |
| span_stacker_bilou_dual_decode_oof | 0129-0512 | 1,296 | 7 | 3 | 10 | 0.77% | 83.72% |
| span_stacker_bilou_dual_decode_oof | 0513-2048 | 2,457 | 5 | 5 | 10 | 0.41% | 89.74% |
| span_stacker_bilou_dual_decode_oof | 2049+ | 1,078 | 2 | 2 | 4 | 0.37% | 86.55% |

## Agreement between audited models

Model order: `mmbert_bilou_viterbi` then `span_stacker_bilou_dual_decode_oof`.

Both models exactly recover 4,529 of 5,379 gold ORG/GEO spans; both assign the same opposite ORG/GEO label on 25 spans.

| First model outcome | Second model outcome | Count |
|---|---|---:|
| exact_correct | exact_correct | 4,529 |
| exact_correct | no_exact_boundary | 84 |
| exact_correct | opposite_label | 1 |
| exact_correct | other_exact_label | 1 |
| no_exact_boundary | exact_correct | 124 |
| no_exact_boundary | no_exact_boundary | 565 |
| no_exact_boundary | opposite_label | 2 |
| no_exact_boundary | other_exact_label | 1 |
| opposite_label | exact_correct | 20 |
| opposite_label | no_exact_boundary | 19 |
| opposite_label | opposite_label | 25 |
| other_exact_label | exact_correct | 2 |
| other_exact_label | no_exact_boundary | 2 |
| other_exact_label | other_exact_label | 4 |

## Source-model consensus and annotation-review candidates

A high-consensus opposite label requires at least 5 of 7 source models (70.00% configured threshold). This flags 8 of 5,379 dev ORG/GEO spans. These are review candidates, not proven annotation errors: sources are correlated and the stacker is derived from them.

| Hash | Doc chars | Gold -> vote | Votes (opp/true) | Audited outputs | Surface | Train ORG/GEO | Signals |
|---|---:|---|---:|---|---|---:|---|
| `01b4b442559c4f8c3c2daabb3d4cd73920260303` | 349 | ORG->GEO | 7/0 | mmbert_bilou_viterbi=GEO, span_stacker_bilou_dual_decode_oof=GEO | Oʻzbekiston | 8/650 | source_consensus_opposes_gold, all_audited_models_predict_opposite, training_surface_majority_is_op… |
| `994d61608980e592907e73a9340589fb20250826` | 290 | GEO->ORG | 7/0 | mmbert_bilou_viterbi=ORG, span_stacker_bilou_dual_decode_oof=ORG | Murad Buildings | 304/3 | source_consensus_opposes_gold, all_audited_models_predict_opposite, training_surface_majority_is_op… |
| `08c2bd62c9ff41fb196c2fd03dfdf77a20260304` | 3,759 | ORG->GEO | 7/0 | mmbert_bilou_viterbi=GEO, span_stacker_bilou_dual_decode_oof=GEO | Buxoro | 9/36 | source_consensus_opposes_gold, all_audited_models_predict_opposite, training_surface_majority_is_op… |
| `c2fec63843bf70ff35959d00c907376220260508` | 683 | ORG->GEO | 7/0 | mmbert_bilou_viterbi=GEO, span_stacker_bilou_dual_decode_oof=GEO | Magic City | 1/12 | source_consensus_opposes_gold, all_audited_models_predict_opposite, training_surface_majority_is_op… |
| `01b4b442559c4f8c3c2daabb3d4cd73920260303` | 349 | ORG->GEO | 7/0 | mmbert_bilou_viterbi=GEO, span_stacker_bilou_dual_decode_oof=GEO | Shimoliy Koreya | 0/10 | source_consensus_opposes_gold, all_audited_models_predict_opposite, training_surface_majority_is_op… |
| `5d0e6695c728d62b` | 791 | ORG->GEO | 5/0 | mmbert_bilou_viterbi=GEO, span_stacker_bilou_dual_decode_oof=GEO | Andijon | 21/54 | source_consensus_opposes_gold, all_audited_models_predict_opposite, training_surface_majority_is_op… |
| `5cb62d35d011e58bfa0cb4a0d2a637d220260301` | 146 | ORG->GEO | 5/0 | mmbert_bilou_viterbi=GEO, span_stacker_bilou_dual_decode_oof=GEO | Rossiya imperiyasining | 0/0 | source_consensus_opposes_gold, all_audited_models_predict_opposite |
| `0170a2db52d1a1018a4fbe0d6760d84620260309` | 62 | GEO->ORG | 5/0 | mmbert_bilou_viterbi=-, span_stacker_bilou_dual_decode_oof=ORG | Oq Uy | 4/0 | source_consensus_opposes_gold, training_surface_majority_is_opposite |

### Candidate contexts

1. `01b4b442559c4f8c3c2daabb3d4cd73920260303` (ORG->GEO, 7/7 opposite votes): **Oʻzbekiston** — isobidagi gʻalabasi bilan yakunlandi. 2026 yilgi Ayollar Osiyo kubogi Shimoliy Koreya 3:0 Oʻzbekiston Gollar: 6,24,41 - Myong Ju-Jong
2. `994d61608980e592907e73a9340589fb20250826` (GEO->ORG, 7/7 opposite votes): **Murad Buildings** — omagazinga kirib sotuvchi qizga: - Singlim mеni xotinim oʻlganiga 3 yil boʻldi, 5 xonalik Murad Buildings domida bitta oʻzim yashayman, zеrikib kеtdim, bitta yaxshi zotdor mushuk bеrsеz, - dеsa S
3. `08c2bd62c9ff41fb196c2fd03dfdf77a20260304` (ORG->GEO, 7/7 opposite votes): **Buxoro** — ov va Shohruh Gadoyev kabi tajribali futbolchilar uch ochko uchun harakat qilishdi, biroq Buxoro himoyasi bardosh berdi. "Lokomotiv" - "Surxon" (3:1) Toshkentdagi "temiryoʻlchilar"ning g
4. `c2fec63843bf70ff35959d00c907376220260508` (ORG->GEO, 7/7 opposite votes): **Magic City** — Xotira va qadrlash kuni munosabati bilan Magic City va Pepsi sizni bayramona shou dasturiga taklif qiladi! Faqat 9, 10 va 11-may kunlari siz
5. `01b4b442559c4f8c3c2daabb3d4cd73920260303` (ORG->GEO, 7/7 opposite votes): **Shimoliy Koreya** — himoliklarning 3:0 hisobidagi gʻalabasi bilan yakunlandi. 2026 yilgi Ayollar Osiyo kubogi Shimoliy Koreya 3:0 Oʻzbekiston Gollar: 6,24,41 - Myong Ju-Jong
6. `5d0e6695c728d62b` (ORG->GEO, 5/7 opposite votes): **Andijon** — rimfinalga yoʻl olishgan edi. Oʻzbekiston Kubogi, 1/4 final. 19 sentabr, Andijon Andijon (Andijon) - Bunyodkor (Toshkent) - 1:2 Gollar: Davron Isoqov (40) - Mirjamol Qosimov (14), Shahzod
7. `5cb62d35d011e58bfa0cb4a0d2a637d220260301` (ORG->GEO, 5/7 opposite votes): **Rossiya imperiyasining** — [22/40] Rossiya imperiyasining Turkiston viloyati gubernatori qaysi javobda TOʻGʻRI koʻrsatilgan? * Kuropatkin * Krijano
8. `0170a2db52d1a1018a4fbe0d6760d84620260309` (GEO->ORG, 5/7 opposite votes): **Oq Uy** — Uzoq kutilgan UFC Oq Uy fayt kardi nega kutilganidek boʻlmadi?

## Reproduction

```bash
python3 scripts/audit_org_geo.py
```

Machine-readable details, including all per-source votes for each listed candidate, are in `audit.json`.
