# Fairness vs. Accuracy in CNNs
## Abstract

This project presents a controlled empirical investigation of the trade-off between accuracy and statistical fairness in convolutional neural networks (CNNs) for medical image classification. The study focuses on chest X-ray data and evaluates the impact of class imbalance in the training set under a strictly controlled experimental protocol.

A fixed DenseNet-121 architecture is used across all experiments, with a balanced and locked evaluation set to ensure reproducibility and comparability. Each experiment varies only the class distribution in the training data, allowing isolation of imbalance effects.

Performance and fairness are evaluated using metrics such as Accuracy, F1-score, Equal Opportunity (ΔTPR), Equalized Odds (ΔFPR), and Disparate Impact (DI). The results enable systematic analysis of how increasing imbalance influences both model performance and fairness disparities between classes.

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
- All training hyperparameters remain fixed across experiments
- Evaluation samples are strictly excluded from the training data
- No rebalancing techniques (e.g., class weighting or oversampling) are applied, in order to isolate the effect of class imbalance.

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
- F1-score (computed per class)
- True Positive Rate (TPR / Recall)
- False Positive Rate (FPR, computed per class)
- Equal Opportunity Gap (ΔTPR)
- Equalized Odds Gap (ΔFPR)
- Disparate Impact (DI)

## Project Structure

```
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
```

## Running Experiments

The experimental protocol consists of executing a series of controlled runs across multiple imbalance scenarios.

Each experiment produces a single result row.
A complete experimental run consists of four scenarios:

- 50-50 (Balanced)
- 60-40 (Mild imbalance)
- 10-90 (High imbalance)
- 1-99 (Extreme imbalance)

### Execution
```bash
cd project
python scripts/run_experiment.py --scenario 50-50
python scripts/run_experiment.py --scenario 60-40
python scripts/run_experiment.py --scenario 10-90
python scripts/run_experiment.py --scenario 1-99
```

Each execution:
- Trains a new model from scratch
- Evaluates on the same locked evaluation set
- Appends one row to the results table

After completing all four runs, the results table will contain one row per scenario, enabling direct comparison.

### Augmentation Experiment
```bash
python experiments/pre_processing/augmentation.py --scenario 50-50
python experiments/pre_processing/augmentation.py --scenario 60-40
python experiments/pre_processing/augmentation.py --scenario 10-90
python experiments/pre_processing/augmentation.py --scenario 1-99
```
This produces an additional set of four rows, corresponding to the augmentation-based pipeline.

## Results

All experiment results are stored as a cumulative CSV file in persistent storage:
`/content/drive/MyDrive/Fairness-vs.-Accuracy-in-CNNs/results_table.csv`

Each experiment appends one row, enabling cumulative analysis across scenarios.

## Fairness–Accuracy Trade-off

The core objective is to analyze how fairness metrics evolve as class imbalance increases.

Rather than assuming fixed outcomes, the study evaluates trends across scenarios:

| Scenario | Expected Behavior |
|----------|------------------|
| 50-50    | Baseline performance with minimal fairness gaps |
| 60-40    | Mild deviation in fairness metrics |
| 10-90    | Noticeable increase in fairness gaps |
| 1-99     | Significant disparity between groups |

Key observations focus on:

- Increase in ΔTPR (Equal Opportunity gap)
- Increase in ΔFPR (Equalized Odds gap)
- Disparate Impact deviating from 1.0

## Visualizations

For each experiment, the following outputs are generated:

- Confusion Matrix (counts and normalized)
- ROC Curve with AUC
- Precision–Recall Curve with AP
- Per-class metrics (Accuracy, TPR, FPR, F1)