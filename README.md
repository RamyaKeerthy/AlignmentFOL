# Assessing the Alignment of FOL Closeness Metrics with Human Judgement

This repository provides a comprehensive study of various metrics and their alignment with human judgment for evaluating First-Order Logic (FOL) closeness.

## Install

1. Clone this Repository
Clone the repository to your local machine:
```commandline
git clone https://github.com/RamyaKeerthy/AlignmentFOL
```

2. Set Up the Environment
Install the required dependencies:
```
pip install -r requirements.txt
```

## Data Generation
To generate data for the evaluation, use the provided Jupyter notebooks within the `notebook` directory. These notebooks contain scripts to create the files essential for replicating the evaluation results presented in the paper.

## Evaluation 
**Perturbation evaluations**
Perturbations can be generated using `notebook/get_perturbations`. Based on the generated perturbations, run the evaluation script to obtain scores for the seven metrics.
```
python run_eval_pert.py
```
**Sample evaluations**
Sample data can be generated using `notebook/get_samples`. Use the following command to run the sample evaluation script:
```
python run_eval_samples.py
```
*Note: Sample generation requires an API key to access the GPT model.*

## Credit
The evaluation code is adapted from [LogicLlama](https://github.com/gblackout/LogicLLaMA.git)

## Licence
This code is licensed under the MIT License and is available for research purposes.

## Citation
If you use this code or reference this work, please cite:
Assessing the Alignment of FOL Closeness Metrics with Human Judgement
