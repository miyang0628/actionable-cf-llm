# Data

Processed modeling tables and prior-pipeline artifacts consumed by the notebooks.
Raw government downloads are **not** redistributed; only processed, de-identified
tables are included.

## Files

| File | Description |
|------|-------------|
| `modeling_dataset_q1_english.csv` | Full modeling table: 4,179 Seoul restaurants (Q1-2022 cohort) × 115 commercial-district features, plus the binary `closure` label. |
| `modeling_dataset_top30.csv` | Same stores restricted to the 30 SHAP-selected features (+ metadata columns). |
| `selected_features.pkl` | List of the 30 selected feature names. |
| `top30_features_list.csv` | The 30 features ranked by mean \|SHAP\| value. |
| `original_outputs/cf_all_experiments.csv` | Unconstrained DiCE counterfactuals from the prior pipeline (2/4/6/8 per instance), used for the actionability audit in notebook 01. |
| `original_outputs/consulting_case{1,2}_changes.csv` | Counterfactual feature changes underlying the two prior case-study strategies. |
| `original_outputs/consulting_case{1,2}_strategy.txt` | LLM-generated consulting strategies evaluated in notebook 03. |

## Source

- National business opening/closure licensing records (public portal).
- Seoul commercial-district statistics: floating population, resident population,
  working population, and estimated sales by district (public portal).

Period: January 2022 – 2023 (seven-quarter observation). Target business type:
restaurants. Outcome: closure within the observation window (closure rate 28.7%).

## Feature actionability taxonomy

The 30 selected features are labeled in `../results/tables/t1_feature_taxonomy.csv`:

- **Actionable (19):** estimated-sales variables (by age, sex, day of week, and
  time slot) — levers a restaurant owner can influence through targeting, menu,
  pricing, promotions, and operating hours.
- **Immutable (11):** resident / working / floating population variables — fixed
  district demographics outside an individual owner's control.
