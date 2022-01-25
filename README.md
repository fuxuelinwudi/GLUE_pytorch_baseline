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
| MNLI  | M acc/MisM acc               | 84.34/84.13 |
| QNLI  | Accuracy                     | 91.20       |
| RTE   | Accuracy                     | 62.82       |
| WNLI  | Accuracy                     | 56.34       |


# hyperparameter setting

| hypers name | value |
|-------|-------|
| lr  | 5e-5 |
| weight decay | 0.01 |
| warmup ratio  | 0.1 |
| epoch | 3 |
| sequence length | 128 |
| seed | 2021 |
