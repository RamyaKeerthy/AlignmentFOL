# Assessing the Alignment of FOL Closeness Metrics with Human Judgement

We present a comprehensive study of existing metrics and their alignment with human judgement on FOL closeness evaluation. 

## Install

1. Clone this repo
```commandline
git clone https://github.com/
```

2. Prepare environment
```
pip install -r requirements.txt
```

## Data Generation

Use `notebook` to generate the files essential for evaluation results mentioned in the paper.

## Evaluation 
**Perturbation evaluations**
The perturbations can be generated using `notebook/get_perturbations`. Based on the perturbation, run the file to extract scores for the seven metrics.
```
python run_eval_pert.py
```

## References
The evaluation code is majorly an adaptation of code from [LogicLlama](https://github.com/gblackout/LogicLLaMA.git)
