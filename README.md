# GLUE_pytorch_baseline


# [GLUE] (https://gluebenchmark.com/tasks)

# Validation support only, unfinished experiments will be added.

| Task  | Metric                       | [BERT-base-uncased](https://huggingface.co/bert-base-uncased)|
|-------|------------------------------|--------|
| CoLA  | Matthews corr                | 60.83 |
| SST-2 | Accuracy                     | 92.78 |
| MRPC  | F1/Accuracy                  | 88.89 |
| STS-B | Pearson/Spearman corr        | 85.87/85.80 |
| QQP   | Accuracy/F1                  | 90.39/87.12 |
| MNLI  | M acc/MisM acc               | 84.34/84.66 |
| QNLI  | Accuracy                     | 91.20 |
| RTE   | Accuracy                     | 67.15 |
| WNLI  | Accuracy                     | 56.34 |


# hyperparameter setting

| task name | learning rate | weight deacay | warmup ratio | epoch | sequence length | seed |
|-------|------|------|-----|---|-----|------|
| CoLA  | 5e-5 | 0.01 | 0.1 | 3 | 128 | 2021 |
| SST-2 | 2e-5 | 0.01 | 0.1 | 3 | 128 | 2021 |
| MRPC | 5e-5 | 0.01 | 0.1 | 5 | 128 | 2021 |
| STS-B | 5e-5 | 0.01 | 0.1 | 4 | 128 | 2021 |
| QQP | 5e-5 | 0.01 | 0.1 | 3 | 128 | 2021 |
| MNLI | 5e-5 | 0.01 | 0.1 | 3 | 128 | 2021 |
| QNLI | 5e-5 | 0.01 | 0.1 | 3 | 128 | 2021 |
| RTE | 5e-5 | 0.01 | 0.1 | 10 | 128 | 2021 |
| WNLI | 5e-5 | 0.01 | 0.1 | 3 | 128 | 2021 |
