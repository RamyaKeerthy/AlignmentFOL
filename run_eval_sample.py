from tqdm import tqdm
from metrics import UniversalMetrics
import fire
import json
import random


def eval_samples(
    data_path="gpt_fol_samples.json"
):
    metric = UniversalMetrics()

    with open(data_path, 'r') as f:
        data = json.load(f)

    outputs = []
    rankbleu = []
    rankle = []
    rankbert = []
    rankrouge = []
    rankmeteor = []
    ranksmatch = []

    for ind, data_point in enumerate(tqdm(data)):
        true_fol = data_point['label']
        bleul = []
        lel=[]
        bertl=[]
        rougel=[]
        meteorl=[]
        smatchppl = []

        for i in [1,2,3]:
            scenario_fol = data_point[f'fol{i}'].split('Rank:')[0].strip()
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
            bleul.append(bleu)
            lel.append(LE)
            bertl.append(bertscore[0])
            rougel.append(rouge)
            meteorl.append(meteor)
            smatchppl.append(Smatchpp_score)
        # Rank
        def rankit(rank_list):
            sorted_numbers = sorted(enumerate(rank_list), key=lambda x: x[1], reverse=True)
            ranks = [0] * len(rank_list)
            current_rank = 1
            i = 0
            while i < len(sorted_numbers):
                j = i
                while j < len(sorted_numbers) and sorted_numbers[j][1] == sorted_numbers[i][1]:
                    j += 1

                rank_group = list(range(current_rank, current_rank + (j - i)))
                random.shuffle(rank_group)
                for k in range(i, j):
                    ranks[sorted_numbers[k][0]] = rank_group[k - i]
                current_rank += (j - i)
                i = j
            return ranks

        rankbleu.append(rankit(bleul))
        rankle.append(LE)
        rankbert.append(bertscore[0])
        rankrouge.append(rouge)
        rankmeteor.append(meteor)
        ranksmatch.append(Smatchpp_score)

        # final_ranks = {'id': data_point['id'],
        #                'text': data_point['premisesNL'],
        #                'label': data_point['label'],
        #                'fol1': data_point['fol1'],
        #                'fol2':data_point['fol2'],
        #                'fol3': data_point['fol3'],
        #                'bleu': (bleul, rankit(bleul)),
        #                'rouge': (rougel, rankit(rougel)),
        #                'meteor': (meteorl, rankit(meteorl)),
        #                'bertscore': (bertl, rankit(bertl)),
        #                'le': (lel, rankit(lel)),
        #                'smatchpp': (smatchppl, rankit(smatchppl)),
        #                }
        # outputs.append(final_ranks)

    return rankbleu, rankle, rankbert, rankrouge, rankmeteor, ranksmatch
    # with open('samples_ranking.json', 'w') as file:
    #     json.dump(outputs, file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    fire.Fire(eval_samples)