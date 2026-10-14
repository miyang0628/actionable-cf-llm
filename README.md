# Actionable by Design: Constraining Counterfactual Explanations and Auditing the Faithfulness of LLM-Generated Business Strategies

Replication and experiment code accompanying a manuscript submitted to an
anonymized journal and currently **under double-blind review**.

> **Anonymity note.** The target journal, authors, affiliations, and funding are
> withheld for double-blind review. This repository contains no
> author-identifying information. When forking or uploading, please avoid
> de-anonymizing through commit metadata (`user.name` / `user.email`).

---

## 1. Motivation

Pipelines for small-business closure support couple a closure classifier with
DiCE counterfactual explanations and a large language model (LLM) that turns the
counterfactuals into consulting strategies. On the data used here the predictor
has a **low performance ceiling** (test AUC ≈ 0.64, maximum predicted closure
probability ≈ 0.60) that the available data cannot raise. The study therefore
moves the contribution away from predictive accuracy and toward the **action
layer**, asking whether the generated guidance is actionable and faithful.

Two coupled contributions:

- **Actionability-constrained counterfactuals.** A feature taxonomy separates
  owner-influenceable levers from exogenous district context; the DiCE search is
  restricted to the influenceable set and compared against the unconstrained
  baseline with confidence intervals and paired significance tests.
- **Faithfulness audit of the LLM layer.** Deterministic, input-grounded metrics
  (grounding coverage, non-actionable leakage, structural completeness) are
  measured across the full matched sample, complemented by an API-gated
  LLM-as-judge ablation with a prompt-only baseline across three model tiers, and
  by a confidence-gating analysis that bounds the trustworthy subset under a weak
  predictor.

---

## 2. Repository structure

```
.
├── data/
│   ├── modeling_dataset_top30.csv        # 4,179 restaurants × 30 selected features (+ meta)
│   ├── modeling_dataset_q1_english.csv   # full 115-feature modeling table
│   ├── selected_features.pkl             # 30 SHAP-selected feature names
│   ├── top30_features_list.csv           # feature ranking by mean |SHAP|
│   └── original_outputs/                 # prior-pipeline artifacts consumed here
│       ├── cf_all_experiments.csv        # unconstrained DiCE CFs (2/4/6/8)
│       └── consulting_case{1,2}_*.{txt,csv}
├── notebooks/
│   ├── 01_feature_taxonomy_and_actionability_audit.ipynb
│   ├── 02_actionability_constrained_dice.ipynb
│   ├── 02b_seed_stability_robustness.ipynb
│   ├── 03_llm_strategy_evaluation.ipynb
│   └── 04_stratified_llm_audit.ipynb
├── results/
│   ├── figures/                          # f1–f8, grayscale, 600 dpi, PNG + PDF
│   └── tables/                           # all result CSVs (committed; see §5)
├── .env.example                          # OpenAI key + model tiers for NB3 PART 5
├── requirements.txt
├── LICENSE
└── README.md
```

Each notebook is a single self-contained code cell with English comments and a
header naming the notebook. The committed notebooks retain their executed outputs,
and `results/` contains every table and figure produced by the runs, so the full
experimental record is included — the notebooks reproduce it deterministically.

---

## 3. Data

The processed modeling tables derive from two public Korean government sources
(Seoul commercial-district statistics and the national business-licensing portal)
for Seoul restaurants newly licensed in 2022 Q1, observed for seven quarters. Raw
downloads are not redistributed; `data/` holds the processed tables the notebooks
consume. The outcome is closure within the observation window (closure rate
28.7%). All column names and categorical values are in English.

---

## 4. How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name python3

jupyter nbconvert --to notebook --execute --inplace notebooks/01_feature_taxonomy_and_actionability_audit.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_actionability_constrained_dice.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02b_seed_stability_robustness.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_llm_strategy_evaluation.ipynb
```

Notebooks 02 and 02b regenerate counterfactuals with DiCE on fixed samples
(`SEED = 44`); each runs in a few minutes. Notebook 03's live LLM scoring
(PART 5) runs only if `OPENAI_API_KEY` is set; all other results are produced
offline. Copy `.env.example` to `.env` and fill in the key and three model tiers:

```
OPENAI_API_KEY=...
LLM_MODEL_WEAK=gpt-4o-mini
LLM_MODEL_MID=gpt-5.4-mini
LLM_MODEL_STRONG=gpt-5.4
```

With these set, `run_llm_evaluation()` runs automatically and scores every
(case × {constrained, unconstrained, prompt-only} × model tier) combination with
a fixed strong-tier judge on the 5-point rubric, writing `t9_llm_judge_scores.csv`.

---

## 5. Notebooks, results, and output tables

**01 — Feature taxonomy and actionability audit.**
Classifies the 30 features (19 influenceable sales / 11 exogenous population; 3 of
the 10 most important by |SHAP| are exogenous). Pooled across settings, 24.5% of
unconstrained counterfactual changes target non-controllable features and 28.3% of
counterfactuals contain at least one such change.
→ `t1_feature_taxonomy.csv`, `t2_actionability_audit.csv`, `f1`–`f3`.

**02 — Actionability-constrained DiCE.**
Matched comparison on 120 closed-and-predicted-to-close restaurants across 2/4/6/8
counterfactuals, with per-store metrics, bootstrap CIs, and paired Wilcoxon tests.
The constrained regime reaches actionability 1.00 (vs. 0.53–0.72) with better
proximity (paired diff +141.1, 95% CI [113.8, 167.7], p≈2e-12) and marginally
higher sparsity (p≈3e-17); diversity and operational-threshold validity do not
differ significantly. Validity is reported at both the native 0.5 boundary
(trivially 1.0) and the operational threshold 0.3142 (≈5–7%). The composite
optimum is 2 counterfactuals, stable across all 8 weightings; the unconstrained
optimum is larger and weight-sensitive.
→ `t3_dice_metrics_by_regime.csv`, `t4_composite_score_by_regime.csv`,
`t5_weight_sensitivity.csv`, `t6_matched_pairwise_stats.csv`,
`cf_generated_sample.csv`, `f4`–`f6`.

**02b — Seed-stability robustness.**
Regenerates at 2 CFs across DiCE seeds {44,1,2,3,4} on a 40-store subsample.
Constrained proximity −25.5±4.9 vs. unconstrained −48.4±51.3 (closer and an order
of magnitude more stable); constrained actionability 1.000±0.000.
→ `t7_seed_stability_robustness.csv`.

**03 — LLM strategy faithfulness.**
Sample-wide input grounding across all 120 matched stores: unconstrained
leaky-counterfactual rate 90.8% with actionable-grounding 0.53–0.72, vs.
constrained 0% / 1.00 (with bootstrap CIs). Two case strategies illustrate the
mechanism (Itaewon non-actionable leakage 1.00). Confidence gating leaves only
164/1,198 (13.7%) confidently predicted closures, 82 (6.8%) at a 0.10 margin.
Includes a three-tier LLM-as-judge harness and a prompt-only baseline (API-gated).
→ `t6_strategy_faithfulness.csv`, `t7_confidence_gating.csv`,
`t8_scaled_input_grounding.csv`, (`t9_llm_judge_scores.csv` when run with a key),
`f7`–`f8`.

**04 — Stratified audit (full consulting-valid population).**
Repeats the counterfactual-basis audit over all 164 closed-and-predicted-to-close
restaurants (7 districts), broken down by district type with bootstrap CIs and
paired Wilcoxon tests. Unconstrained actionable-grounding 0.593 (Developed
Commercial Area, leaky 88.1%) and 0.319 (Special Tourist Zone, leaky 100%) vs.
1.000 / 0% constrained (p≈9e-25 and 8e-07). Reports the honest coverage limit (a
weak predictor confidently flags closures in only two of four district types).
Includes a stratified LLM generation + judge loop (3 conditions × 3 tiers,
API-gated). → `t9_grounding_by_district.csv`, `t10_grounding_by_type.csv`,
`t11_grounding_paired_by_type.csv`, (`t12_output_audit_by_condition.csv` with key),
`f9`–`f10` (`f11` with key).

---

## 6. Reproducibility notes

- Randomness is seeded (`SEED = 44`; split `random_state = 42`).
- The closure model (XGBoost on 30 features) is retrained inside each notebook, so
  no pickled estimator is loaded across library versions.
- Matched comparisons report 95% bootstrap CIs (2000 resamples) and two-sided
  paired Wilcoxon signed-rank tests; seed-stability over 5 DiCE seeds.
- DiCE post-hoc sparsity optimization is disabled for tractable runtime (enabling
  it is computationally prohibitive at this scale); sparsity is measured directly,
  and the proximity advantage is structural (the constrained search varies a
  strict subset of features).
- All tables and figures under `results/` regenerate deterministically.

---

## 7. License

Code released under the MIT License (see `LICENSE`). Government source data remain
subject to their original providers' terms.
