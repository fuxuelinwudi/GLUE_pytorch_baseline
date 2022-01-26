# GLUE_pytorch_baseline


# [GLUE] (https://gluebenchmark.com/tasks)

# Validation support only, unfinished experiments will be added.

| Task  | Metric                       | [BERT-base-uncased](https://huggingface.co/bert-base-uncased)|
|-------|------------------------------|-------------|
| CoLA  | Matthews corr                | 60.83       |
| SST-2 | Accuracy                     | 85.56       |
| MRPC  | F1/Accuracy                  | 87.78       |
| STS-B | Pearson/Spearman corr        | 85.76       |
| QQP   | Accuracy/F1                  | 90.39       |
| MNLI  | M acc/MisM acc               | 84.34/84.66 |
| QNLI  | Accuracy                     | 91.20       |
| RTE   | Accuracy                     | 62.82       |
| WNLI  | Accuracy                     | 56.34       |


# hyperparameter setting

| task name | learning rate | weight deacay | warmup ratio | epoch | sequence length | seed |
|-------|------|------|-----|---|-----|------|
| CoLA  | 5e-5 | 0.01 | 0.1 | 3 | 128 | 2021 |
| SST-2 | 5e-5 | 0.01 | 0.1 | 3 | 128 | 2021 |
| MRPC | 5e-5 | 0.01 | 0.1 | 3 | 128 | 2021 |
| STS-B | 5e-5 | 0.01 | 0.1 | 3 | 128 | 2021 |
| QQP | 5e-5 | 0.01 | 0.1 | 3 | 128 | 2021 |
| MNLI | 5e-5 | 0.01 | 0.1 | 3 | 128 | 2021 |
| QNLI | 5e-5 | 0.01 | 0.1 | 3 | 128 | 2021 |
| RTE | 5e-5 | 0.01 | 0.1 | 3 | 128 | 2021 |
| WNLI | 5e-5 | 0.01 | 0.1 | 3 | 128 | 2021 |
