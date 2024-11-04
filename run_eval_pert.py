import numpy as np
from tqdm import tqdm
from metrics import UniversalMetrics
import fire
import json


def eval_pert(
    data_path="perturbations.json",
    perturbation = "remove_negation"
):
    metric = UniversalMetrics()

    with open(data_path, 'r') as f:
        data = json.load(f)

    Bleu_final = []
    LE_final = []
    Rouge_final = []
    BertScore_final = []
    Meteor_final = []
    Smatchpp_final = []

    for ind, data_point in enumerate(tqdm(data)):
        true_fol = data_point['premisesFOL']
        scenario_fol = data_point[perturbation] # perturbations
        res = metric.evaluate(
            None,
            true_fol,
            None,
            scenario_fol
        )
        bleu, LE, rouge, bertscore, meteor, smatchpp = res.FOL_bleu, res.FOL_LE, res.FOL_Rouge, res.FOL_BertScore, res.FOL_Meteor, res.FOL_Smatchpp

        if isinstance(smatchpp, float):
            Smatchpp_score = smatchpp
        else:
            Smatchpp_score = 0

        Bleu_score = bleu
        LE_score = LE
        Rouge_score = rouge
        Bert_Score = bertscore
        Meteor_score = meteor


        Bleu_final.append(Bleu_score)
        LE_final.append(LE_score)
        Rouge_final.append(Rouge_score)
        BertScore_final.append(Bert_Score)
        Meteor_final.append(Meteor_score)
        Smatchpp_final.append(Smatchpp_score)


    print("Bleu Score", np.mean(Bleu_final))
    print("LE score", np.mean(LE_final))
    print("Rouge score", np.mean(Rouge_final))
    print("Bert score", np.mean(BertScore_final))
    print("Meteor score", np.mean(Meteor_final))
    print("Smatchpp score", np.mean(Smatchpp_final))


if __name__ == '__main__':
    fire.Fire(eval_pert)