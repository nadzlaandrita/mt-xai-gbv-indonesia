# Multi-target explainable AI for province-level gender-based violence in Indonesia

Code and data for the article *Multi-Target Explainable AI for Province-Level Gender-Based Violence in
Indonesia: Explaining Sexual and Non-Sexual Violence with Socioeconomic Indicators* (submitted to
PeerJ Computer Science).

The study models recorded rates of sexual and non-sexual violence per 100,000 population in 34
Indonesian provinces (2021 to 2024) from 23 socioeconomic indicators. Random Forest, CatBoost and
ElasticNet are fitted in multi-target and single-target form, explained with SHAP and LIME, and the
explanations are tested for cross-target consistency, stability and fidelity. A province-level
bootstrap and synthetic data with known drivers show that multi-target models inflate the similarity
of explanations across targets.

## Repository structure

```
data/
  raw/raw_dataset.csv           compiled raw panel (same content as RAW_dataset.xlsx)
  processed/dataset_bersih.csv  cleaned panel produced by src/pipeline.py
  DATA_DICTIONARY.md            variables, sources and cleaning steps
src/
  pipeline.py                   cleaning, tuning, models, SHAP, LIME, stability, aggregation
  validate.py                   single- vs multi-target comparison, bootstrap, synthetic data
  run_validate.py               entry point for validate.py
  figures.py                    Figures 3 to 6 of the paper
notebooks/
  Multi_Target_XAI.ipynb        the full analysis in one notebook (sections 1 to 9, comments in Indonesian).
                                Section 6 exports files for a companion web dashboard that is not part of the paper.
results/
  main/                         outputs of src/pipeline.py
  validation/                   outputs of src/run_validate.py
figures/                        figures used in the paper
```

## How to reproduce

The results were produced with Python 3.13 and the package versions pinned in `requirements.txt`.

```bash
pip install -r requirements.txt

# main analysis (about 5 minutes)
RAW_PATH=data/raw/raw_dataset.csv OUT=results/main python src/pipeline.py

# validation experiments (slower, 200 bootstrap resamples and the synthetic study)
RAW_PATH=data/raw/raw_dataset.csv OUT=results/validation python src/run_validate.py

# figures
python src/figures.py
```

`N_TRIALS` (Optuna trials, default 30) and `N_BOOT` (bootstrap resamples, default 200) can be set as
environment variables for a quicker test run. All random processes use seed 67.

The hyperparameters of the multi-target Random Forest are fixed in `src/pipeline.py`
(`n_estimators=400, max_depth=10, min_samples_leaf=1, max_features='sqrt'`) because two candidate
settings reached almost the same cross-validation score but different explanations. This is discussed
in the paper.

## Result files

File names follow the original (Indonesian) analysis.

| File | Content | Paper |
|---|---|---|
| `results/main/kinerja.csv` | R², MAE, RMSE on the 2024 test year, all models and baselines | Table 4 |
| `results/main/loo_provinsi.csv` | Leave-one-province-out performance | Results |
| `results/main/korelasi_spearman.csv` | Spearman correlation of each indicator with the targets | Results |
| `results/main/shap_global_multi.csv`, `shap_global_single.csv` | Normalised global SHAP importance, Equation (3) | Figure 3 |
| `results/main/shap_multi.npy` | SHAP values of the multi-target Random Forest | Figure 4 |
| `results/main/lintas_target.json` | Cross-target consistency per province | Table 5 |
| `results/main/stabilitas_shap.csv` | Stability across five seeds | Results |
| `results/main/lime_fidelity.csv` | LIME local R² and agreement with SHAP | Results |
| `results/main/provinsi.json` | Composite score, level and dominant factors per province | Table 8 |
| `results/main/parameter.csv` | Selected hyperparameters | Methods |
| `results/validation/single_vs_multi.csv` | Multi- vs single-target performance and similarity | Table 5 |
| `results/validation/bootstrap_ringkasan.csv` | Bootstrap summary (`selisih` = difference, `CI_bawah`/`CI_atas` = 95% interval) | Table 6, Figure 5 |
| `results/validation/sintetis_ringkasan.csv` | Synthetic experiment summary (`Sama`/`Sebagian`/`Berbeda` = shared/partial/distinct) | Table 7, Figure 6 |
| `results/validation/cek_skala_target.csv` | Check with standardised targets | Results |

Target names: `rate_seksual` = sexual violence rate, `rate_non_seksual` = non-sexual violence rate.

## Data sources

* Victims of violence, SIMFONI PPA, Ministry of Women's Empowerment and Child Protection (KemenPPPA),
  https://kekerasan.kemenpppa.go.id/ringkasan
* Socioeconomic indicators, Statistics Indonesia (BPS), https://www.bps.go.id

See `data/DATA_DICTIONARY.md` for each variable.

## Licence

Code under the MIT licence (`LICENSE`). Data and results under CC BY 4.0 (`LICENSE-DATA.md`).

## Citation

See `CITATION.cff`. The article DOI will be added after publication.
