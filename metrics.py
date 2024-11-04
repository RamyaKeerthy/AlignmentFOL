from dataclasses import dataclass
from typing import List, Optional
import evaluate
from fol_parser import VecRuleEvaluator, parse_text_FOL_to_tree, msplit
from smatchpp import Smatchpp
from graph_parser import parse_tree_to_amr

@dataclass
class MetricsOutput:
    FOL_bleu: float
    FOL_LE: float
    FOL_Rouge: float
    FOL_BertScore: float
    FOL_Meteor: float
    FOL_Smatchpp: float
    FOL_true_inputs: Optional[List[str]]
    FOL_binded_pred_inputs: Optional[List[str]]


@dataclass
class UniversalMetricsOutput(MetricsOutput):
    instruct_bleu: float


class Metrics:

    def __init__(self):
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        self.meteor = evaluate.load('meteor')
        self.FOL_tokenizer = lambda x: msplit(x)[0]

    def compute_FOL_bleu(self, pred_seq: str, true_seq: str):
        min_len = min(map(lambda x: len(self.FOL_tokenizer(x)), [pred_seq, true_seq]))
        res = self.bleu.compute(predictions=[pred_seq], references=[[true_seq]],
                                tokenizer=self.FOL_tokenizer, max_order=min(4, min_len))
        return res['bleu']

    def compute_FOL_rouge(self, pred_seq: str, true_seq: str):
        min_len = min(map(lambda x: len(self.FOL_tokenizer(x)), [pred_seq, true_seq]))
        res = self.rouge.compute(predictions=[pred_seq], references=[[true_seq]],
                                 tokenizer=self.FOL_tokenizer)
        return res['rougeL']

    def compute_FOL_bertscore(self, pred_seq: str, true_seq: str):
        min_len = min(map(lambda x: len(self.FOL_tokenizer(x)), [pred_seq, true_seq]))
        res = self.bertscore.compute(predictions=[pred_seq], references=[[true_seq]],
                                     lang="en")
        return res['f1']

    def compute_FOL_meteor(self, pred_seq: str, true_seq: str):
        min_len = min(map(lambda x: len(self.FOL_tokenizer(x)), [pred_seq, true_seq]))
        res = self.meteor.compute(predictions=[pred_seq], references=[[true_seq]])
        return res['meteor']

    def compute_instruct_bleu(self, pred_seq: str, true_seq: str):
        min_len = min(map(lambda x: len(x.split()), [pred_seq, true_seq]))
        res = self.bleu.compute(predictions=[pred_seq], references=[[true_seq]],
                                max_order=min(4, min_len))
        return res['bleu']

    def compute_LE(self, pred_text_FOL: str, true_text_FOL: str):
        true_root, pred_root = parse_text_FOL_to_tree(true_text_FOL), parse_text_FOL_to_tree(pred_text_FOL)

        # parsing true FOL should never fail
        # assert true_root is not None, 'failed parsing true text FOL %s' % true_text_FOL
        if true_root is None:
            return 0, "", ""

        # parsing pred FOL can fail if model produces invalid rule, in which case, LE score is 0
        if pred_root is None:
            return 0., None, None

        # if both parsed successfully, then compute LE score
        score, true_inputs, binded_pred_inputs = \
            VecRuleEvaluator.find_best_LE_score(
                true_root,
                pred_root,
                soft_binding=True,
                greedy_match=True,
                top_n=1000
            )
        return score, true_inputs, binded_pred_inputs

    def compute_smatchpp(self, pred_text_FOL: str, true_text_FOL: str):
        true_root, pred_root = parse_text_FOL_to_tree(true_text_FOL), parse_text_FOL_to_tree(pred_text_FOL)

        if true_root is None:
            return 0, "", ""

        if pred_root is None:
            return 0., None, None

        true_root_amr, pred_root_amr = parse_tree_to_amr(true_root), parse_tree_to_amr(pred_root)
        measure = Smatchpp()
        score = measure.score_pair(true_root_amr, pred_root_amr) # {'main': {'F1': 100.0, 'Precision': 100.0, 'Recall': 100.0}}
        return score['main']['F1']

    def evaluate(self, pred_seq: str, true_seq: str):
        raise NotImplementedError


class UniversalMetrics(Metrics):

    def evaluate(self, orig_comments: Optional[str], orig_fol: str, pred_comments: str, pred_fol: str):
        # FOL_bleu = self.compute_instruct_bleu(pred_fol, orig_fol)
        FOL_bleu = self.compute_FOL_bleu(pred_fol, orig_fol)
        FOL_eval = self.compute_LE(pred_fol, orig_fol)
        FOL_rouge = self.compute_FOL_rouge(pred_fol, orig_fol)
        FOL_bertscore = self.compute_FOL_bertscore(pred_fol, orig_fol)
        FOL_meteor = self.compute_FOL_meteor(pred_fol, orig_fol)
        FOL_smatchpp = self.compute_smatchpp(pred_fol, orig_fol)

        return UniversalMetricsOutput(
            FOL_bleu=FOL_bleu,
            FOL_Rouge=FOL_rouge,
            FOL_BertScore=FOL_bertscore,
            FOL_Meteor=FOL_meteor,
            FOL_Smatchpp=FOL_smatchpp,
            FOL_LE=FOL_eval[0],
            FOL_true_inputs=FOL_eval[1],
            FOL_binded_pred_inputs=FOL_eval[2]
        )


