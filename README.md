# Fairness vs. Accuracy in CNNs

This project presents an empirical study examining the trade-off between model performance (Accuracy) and fairness in convolutional neural networks (CNNs), using chest X-ray images for binary classification (NORMAL vs. PNEUMONIA).

## Research Objective

The study aims to analyze how class imbalance in training data affects:

- Overall model performance
- Per-class performance
- Statistical fairness metrics
- The trade-off between accuracy and fairness

## Methodology

- Fixed model architecture: DenseNet-121
- Transfer Learning with ImageNet pre-trained weights
- Adaptation to grayscale input (single channel)
- Fixed and balanced evaluation set (1500 NORMAL, 1500 PNEUMONIA)
- Each experiment trains a new model from scratch
- No use of checkpoints or model reuse
- Training configuration remains constant across experiments

## Experimental Scenarios

| Scenario | PNEUMONIA | NORMAL |
|----------|----------|--------|
| 50-50    | 1500     | 1500   |
| 60-40    | 1500     | 1000   |
| 10-90    | 1500     | 166    |
| 1-99     | 1500     | 15     |

Each scenario represents a different level of class imbalance in the training data.

## Evaluation Metrics

The following metrics are computed:

- Accuracy (overall and per class)
- F1-score (per class)
- True Positive Rate (TPR / Recall)
- False Positive Rate (FPR)
- Equal Opportunity Gap (ΔTPR)
- Equalized Odds Gap (ΔFPR)
- Disparate Impact (DI)

In addition, the following visualizations are generated:

- Confusion Matrix (counts and normalized)
- ROC Curve (with AUC)
- Precision–Recall Curve (with AP)
- Per-class performance comparison

## Project Structure

project/
  common/
    config.py
    dataset.py
    model.py
    utils.py
    experiment_logger.py
    pipeline/
      train.py
      eval.py

  scripts/
    scenarios.py
    run_experiment.py

  experiments/
    pre_processing/
      augmentation.py

## Running Experiments

To execute a baseline experiment:

```bash
cd project
python scripts/run_experiment.py --scenario 50-50
python scripts/run_experiment.py --scenario 60-40
python scripts/run_experiment.py --scenario 10-90
python scripts/run_experiment.py --scenario 1-99

### Augmentation Experiment
python experiments/pre_processing/augmentation.py --scenario 50-50
```