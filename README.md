# Training-Label Horizon Optimisation

Can the best training-label horizon differ from the horizon we want to predict?

This project explores that question using **LightGBM**, **Optuna**, and a synthetic asset panel. It is inspired by [The Label Horizon Paradox: Rethinking Supervision Targets in Financial Forecasting](https://proceedings.mlr.press/v306/song26l.html).

## Overview

The prediction goal remains fixed at **horizon h = 10**. We compare models trained on:

- The final target horizon, h = 10.
- Each earlier horizon separately.
- A learned mixture of training-label horizons.

The synthetic data combines a signal that gradually saturates with noise that continues accumulating. This creates a setting where earlier labels may provide cleaner supervision for predicting the final target.

## Methodology

- **Dataset:** 240 dates, 35 assets, 24 features, and return labels at horizons 1–10.
- **Model:** LightGBM regression using cross-sectionally standardised training labels.
- **Validation:** Chronological train/validation/test splits with 10-date purge gaps.
- **Optimisation:** 30 Optuna/TPE trials selecting the centre and width of a distribution of horizon weights.
- **Outer objective:** Mean daily validation Pearson IC at h = 10, with a normalised entropy reward.
- **Evaluation:** Mean daily cross-sectional Pearson IC and Spearman rank IC against h = 10.

After selecting the horizon weights using validation data, the final models are refitted on the development data and evaluated on the held-out test period.

## Results

Stored notebook results using random seed **42**:

| Training approach | Test Pearson IC | Test rank IC |
| :--- | ---: | ---: |
| Direct target: h = 10 | 0.3782 | 0.3446 |
| Selected horizon: h = 3, retrained | **0.3897** | **0.3577** |
| Learned weighted-label mixture | 0.3887 | 0.3572 |
| Equal average of selected horizons 3, 4 and 2 | 0.3866 | 0.3574 |

Retraining on the selected horizon improved test Pearson IC by **0.0115** and rank IC by **0.0131** relative to training directly on h = 10.

The weighted-label mixture is retained as a diagnostic comparison. Exact results may vary with package versions.

## Getting Started

No external dataset or API credentials are required. The notebook generates the synthetic data.

Install the dependencies:

```bash
python -m pip install lightgbm optuna numpy pandas scipy matplotlib ipython jupyter
```

Open the notebook:

```bash
jupyter notebook "Bilevel_Horizon_Optimization_LightGBM.ipynb"
```

Run all cells from top to bottom to generate the data, compare training horizons, optimise the horizon weights, and display the final results.

## Relationship to the Paper

This project is an **independent adaptation**, rather than a reproduction of the paper's full experiments.

The paper uses gradient-based bilevel optimisation. Here, LightGBM is treated as a black box, and Optuna/TPE performs the outer search over training-label horizon weights.

The final comparisons include retraining on the largest-weight horizon and averaging the three largest-weight labels.

## Limitations and Extensions

The experiment uses synthetic data designed to contain a signal–noise trade-off. Its results do not establish that earlier training labels generally improve real-market forecasts. IC measures predictive correlation, rather than trading profitability.

Possible extensions include:

- Testing stability across random seeds.
- Varying signal-realisation and noise-accumulation rates.
- Applying the procedure to real asset data.
- Using expanding-window validation, with horizon selection confined to past data.

## Reference

Chen-Hui Song, Shuoling Liu and Liyuan Chen (2026).

**The Label Horizon Paradox: Rethinking Supervision Targets in Financial Forecasting.** ICML 2026.

[Paper](https://proceedings.mlr.press/v306/song26l.html) · [Authors' Open-Source Demo](https://github.com/EFundAI/label-horizon-paradox)
