# BanglaBERT + XGBoost Ensemble

- Device: `cuda`
- BanglaBERT test macro F1: `0.7040`
- XGBoost test macro F1: `0.7130`
- Ensemble alpha for BanglaBERT: `0.45`
- Ensemble test macro F1: `0.7159`
- Stacking ensemble test macro F1: `0.7038`

## Notes

- XGBoost uses BanglaBERT CLS embeddings, BanglaBERT probabilities, confidence features, simple text features, and category one-hot features.
- Weighted ensemble uses validation-tuned alpha.
- Stacking ensemble learns a logistic-regression combiner from validation-set branch outputs.
