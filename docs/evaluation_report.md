# Evaluation report

Embedder: **deterministic mock**. Threshold: **0.8**. Generation: fake (no completion spend).

> The mock embedder is a lexical hash with a tiny synonym list. These numbers test the **eligibility/safety pipeline**. They are **not** a
> measurement of OpenAI embedding quality or of production hit rates. Re-run with `--live` for real embedding scores.

## Summary

- Cases: 28; outcomes matching expectation: 28/28
- Paraphrase hit rate (should hit): 100%
- Unsafe hits with the full pipeline: **0** of 20 must-not-hit cases
- Unsafe hits a similarity-only cache would serve (same threshold, no signature/version/personal checks): **8**

## Threshold sweep

| threshold | paraphrase hit rate | unsafe hits (pipeline) | unsafe hits (similarity only) |
|---|---|---|---|
| 0.6 | 100% | 0 | 10 |
| 0.7 | 100% | 0 | 10 |
| 0.8 | 100% | 0 | 8 |
| 0.9 | 88% | 0 | 0 |
| 0.95 | 75% | 0 | 0 |

## Per case

| id | kind | score | expected | got | status |
|---|---|---|---|---|---|
| p01 | paraphrase | 0.9186 | hit | hit | semantic_hit |
| p02 | paraphrase | 1.0 | hit | hit | semantic_hit |
| p03 | paraphrase | 1.0 | hit | hit | semantic_hit |
| p04 | paraphrase | 1.0 | hit | hit | semantic_hit |
| p05 | paraphrase | 1.0 | hit | hit | semantic_hit |
| p06 | paraphrase | 1.0 | hit | hit | semantic_hit |
| p07 | paraphrase | 0.866 | hit | hit | semantic_hit |
| p08 | paraphrase | 1.0 | hit | hit | semantic_hit |
| n01 | number_change | 0.8148 | miss | miss | miss |
| n02 | number_change | 0.8571 | miss | miss | miss |
| n03 | number_change | 0.75 | miss | miss | miss |
| n04 | number_change | 0.8333 | miss | miss | miss |
| g01 | negation | 0.8165 | miss | miss | miss |
| g02 | negation | 0.866 | miss | miss | miss |
| g03 | negation | 0.866 | miss | miss | miss |
| i01 | identifier | 0.75 | ineligible | ineligible | ineligible |
| i02 | identifier | 0.8529 | ineligible | ineligible | ineligible |
| q01 | quoted | 0.8333 | miss | miss | miss |
| v01 | context_version | 1.0 | miss | miss | miss |
| v02 | template_version | 1.0 | miss | miss | miss |
| v03 | param_change | 1.0 | miss | miss | miss |
| x01 | personal | 0.0 | ineligible | ineligible | ineligible |
| x02 | personal | 0.6149 | ineligible | ineligible | ineligible |
| x03 | personal | 1.0 | ineligible | ineligible | ineligible |
| u01 | unrelated | 0.0 | miss | miss | miss |
| u02 | unrelated | 0.0 | miss | miss | miss |
| u03 | unrelated | 0.1414 | miss | miss | miss |
| x04 | personal | 1.0 | ineligible | ineligible | ineligible |

## Cost comparison (a model, not a measurement)

Assumptions: 10000 requests, 200 input + 150 output tokens per generation, 30 embedding tokens per request, prices from settings defaults (placeholders).
The hit rate used (100%) is the paraphrase hit rate above applied to ALL traffic, which is optimistic: real traffic has a different mix.

| | USD |
|---|---|
| no cache | 1.2000 |
| with cache | 0.0060 |
| estimated net savings | 1.1940 |

Break-even hit rate (embedding cost per request / generation cost per request): 0.50%. Below this the cache costs more than it saves.

## Known gaps

- Small hand-written dataset; real traffic will differ.
- Signature checks are heuristics: paraphrased numbers ("thirty" vs "30") or negation by antonym are not detected.
- No measurement of answer quality for semantic hits beyond the rules above.
