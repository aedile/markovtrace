# E2 Recommendation: Shapley sampling strategy for markovtrace v1.0

**Status:** Draft (empirical sections pending execution).
**Date:** 2026-05-14
**Branch:** spike/E2-shapley-sampling

## Context

E2 compares TokenSHAP, naive Monte Carlo Shapley, and KernelSHAP on a
handcrafted 9-prompt set (1k/4k/16k tokens by RAG, instruction-following,
and reasoning categories), measuring runtime, faithfulness (sufficiency,
comprehensiveness, removal-as-prediction-change per DeYoung et al. 2020),
variance across at least 5 repeats, and token spend. The model under
test is Anthropic Haiku 4-5 at temperature 0 with disk caching. The
total spend cap for the notebook is $50, with a $45 abort threshold to
leave a 10% buffer.

## Methods compared

1. **TokenSHAP** (Horovicz & Goldshmidt 2024, arXiv:2407.10114): token-level
   adaptation of Shapley with internal Monte Carlo sampling controlled by
   a `sampling_ratio` parameter.
2. **Naive Monte Carlo Shapley** (Strumbelj & Kononenko 2014; Castro et al.
   2009): uniform permutation sampling with marginal-contribution
   accumulation. Implemented from scratch in `helpers/shapley.py`.
3. **KernelSHAP adapted to tokens** (Lundberg & Lee 2017): binary-feature
   treatment per token, weighted regression on coalition-mask samples.
   Implemented as an adapter over the `shap` library in
   `helpers/kernel_shap.py`.

## Faithfulness metrics

Per DeYoung et al. 2020, Section 4.1:

- **Sufficiency** (Equation 1): `p(y | x) - p(y | rationale)` where the
  rationale is the top-k attributed tokens. Lower is better.
- **Comprehensiveness** (Equation 2): `p(y | x) - p(y | x \ rationale)`.
  Higher is better.
- **Removal-as-prediction-change**: average L1 distance between the
  full-prompt and single-token-ablated next-token distributions. The
  notebook documents the choice of L1 over KL and over argmax-flip; L1
  is symmetric, bounded, and does not require exponentiating log-probs.

## Results

The notebook records all measurements with 95% bootstrap confidence
intervals from at least 5 independent repeats per cell. The table below
gets populated by the PM execution pass.

| Method | Prompt size | Runtime (s) | Sufficiency | Comprehensiveness | RaPC | $ per run |
|---|---|---|---|---|---|---|
| TokenSHAP | 1k | pending | pending | pending | pending | pending |
| TokenSHAP | 4k | pending | pending | pending | pending | pending |
| TokenSHAP | 16k | pending | pending | pending | pending | pending |
| Naive MC | 1k | pending | pending | pending | pending | pending |
| Naive MC | 4k | pending | pending | pending | pending | pending |
| Naive MC | 16k | pending | pending | pending | pending | pending |
| KernelSHAP | 1k | pending | pending | pending | pending | pending |
| KernelSHAP | 4k | pending | pending | pending | pending | pending |
| KernelSHAP | 16k | pending | pending | pending | pending | pending |

## Recommendation

**Pending data.** The recommendation will name a single method based on
the following decision rule, applied in order:

1. **Faithfulness floor.** The method must outperform a random-attribution
   baseline on at least two of the three faithfulness metrics, with the
   improvement larger than the 95% CI width.
2. **Runtime envelope.** Total wall-clock on a 4k-token prompt at the
   recommended sample budget must stay under 60 seconds.
3. **Cost envelope.** USD per attribution call at the recommended budget
   must stay under $0.10 on Haiku 4-5.
4. **Implementation cost.** If two methods clear gates 1 through 3,
   prefer the one whose implementation is shorter or whose upstream
   dependency is more stable.

Pre-execution sketch: TokenSHAP is the most likely v1.0 default. The
upstream library is purpose-built for the task and is actively
maintained on a public GitHub. Naive Monte Carlo is the implementation
baseline that markovtrace ships internally; it serves as the fallback if
TokenSHAP becomes unmaintained or its API drifts. KernelSHAP is the
most expensive of the three at typical prompt sizes because each
coalition sample triggers a full API call; the spike confirms or
falsifies this empirically.

The notebook also reports per-method runtime scaling against prompt
length. If KernelSHAP scales worse than linearly while the others stay
linear, that fact lands in the recommendation rationale.

## Generalization scope

This recommendation applies to the spike's test set: 9 handcrafted
prompts spanning 1k/4k/16k tokens and RAG/instruction/reasoning
categories, scored against Anthropic Haiku 4-5 at temperature 0. The
recommendation does not claim theoretical optimality, only the best
measured tradeoff on this test set under the $50 spend cap. The v1.0
implementation should retain the comparison harness so the choice can
be re-validated on new prompt sets.

## Risks

- **TokenSHAP API drift.** The upstream repo has no published release
  tags as of the spike date (2026-05-14). The notebook pins the
  install command to a specific commit SHA captured at notebook
  execution; v1.0 inherits that pin.
- **Anthropic Haiku 4-5 throughput limits.** Burst attribution under
  load may saturate rate limits. v1.0 must implement exponential backoff
  in the adapter layer; this is tracked by E3 acceptance criteria.
- **Faithfulness top-k default.** DeYoung et al. 2020 leaves k as a
  hyperparameter. The notebook reports results at k=10%, k=25%, and
  k=50%. The recommendation will name a default k for v1.0.
- **API non-determinism.** The notebook characterizes the variance
  floor by running an identical prompt 10 times at temperature 0
  before attributing variance to any estimator. That measurement
  appears in the variance pass.
- **KernelSHAP adaptation soundness.** Treating tokens as independent
  binary features is a known approximation. If KernelSHAP underperforms
  on the spike's prompts, the recommendation explicitly does not
  conclude that KernelSHAP is wrong; it concludes that the cheap
  token-binary adaptation is wrong. A future spike can revisit with a
  sentence-level mask.

## Open questions for PM

1. **Sample budget for v1.0.** The notebook tests budgets at 10, 50,
   200, and 1000. v1.0 should ship one default; pre-execution intuition
   is 200 for typical 1k-4k prompts.
2. **Top-k default.** Pin a single default in v1.0, or expose k as a
   user-facing parameter? Recommendation defers to PM after data.
3. **Promotion path for helpers.** The spike's `helpers/` modules are
   not under `src/markovtrace/`. CLAUDE.md Rule 9 and Rule 10 require
   tests before any promotion. The recommendation notes the cost of
   promotion so PM can plan the work in E6.

## References

- Castro, J., Gomez, D., & Tejada, J. (2009). Polynomial calculation of
  the Shapley value based on sampling. Computers & Operations Research,
  36(5), 1726-1730. https://doi.org/10.1016/j.cor.2008.04.004 (Accessed
  2026-05-14).
- DeYoung, J., Jain, S., Rajani, N. F., Lehman, E., Xiong, C., Socher,
  R., & Wallace, B. C. (2020). ERASER: A Benchmark to Evaluate
  Rationalized NLP Models. ACL 2020.
  https://aclanthology.org/2020.acl-main.408/ (Accessed 2026-05-14).
- Horovicz, M., & Goldshmidt, R. (2024). TokenSHAP: Interpreting Large
  Language Models with Monte Carlo Shapley Value Estimation. arXiv
  preprint arXiv:2407.10114. https://arxiv.org/abs/2407.10114 (Accessed
  2026-05-14).
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to
  Interpreting Model Predictions. Advances in Neural Information
  Processing Systems 30.
  https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
  (Accessed 2026-05-14).
- Strumbelj, E., & Kononenko, I. (2014). Explaining prediction models
  and individual predictions with feature contributions. Knowledge and
  Information Systems, 41(3), 647-665.
  https://doi.org/10.1007/s10115-013-0679-x (Accessed 2026-05-14).
