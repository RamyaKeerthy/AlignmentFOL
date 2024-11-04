import re
import nltk
from copy import deepcopy
import numpy as np
from itertools import product, permutations
from typing import List
from Levenshtein import distance as edit_dist
import concurrent.futures

op_ls = ['⊕', '∨', '∧', '→', '↔', '∀', '∃', '¬', '(', ')', ',', '=']

sym_reg = re.compile(r'[^⊕∨∧→↔∀∃¬(),>&=]+')

cfg_template = """
S -> F | Q F | '¬' S | '(' S ')'
Q -> QUANT VAR | QUANT VAR Q
F -> '¬' '(' F ')' | '(' F ')' | F OP F | L 
OP -> '⊕' | '∨' | '∧' | '→' | '↔'
L -> '¬' PRED '(' TERMS ')' | PRED '(' TERMS ')' | VAR '=' VAR | '¬' VAR '=' VAR
TERMS -> TERM | TERM ',' TERMS
TERM -> CONST | VAR
QUANT -> '∀' | '∃'
"""

# used in perturbation
last_nt_nodes = set(['PRED', 'OP', 'CONST', 'VAR', 'QUANT'])
# used in node insertion
insertable_nt_nodes = set(['Q', 'S', 'TERMS', 'F'])
# used in node deletion
deleteable_nt_nodes = set(['Q', 'TERMS', 'F', 'L'])


def parse_text_FOL_to_tree(rule_str):
    """
        Parse a text FOL rule into nltk.tree

        Returns: nltk.tree, or None if the parse fails
    """
    rule_str = reorder_quantifiers(rule_str)

    r, parsed_fol_str = msplit(rule_str)

    cfg_str = make_cfg_str(r)

    grammar = nltk.CFG.fromstring(cfg_str)
    if len(grammar.productions()) > 80:
        return None

    def parse_with_timeout(grammar, r):
        parser = nltk.ChartParser(grammar)
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(parser.parse_one, r)
                result_tree = future.result(timeout=5)  # Set the timeout value in seconds
                return result_tree
        except Exception as e:
            print("exception:", e)
            return None  # Handle the timeout by returning None

    tree = parse_with_timeout(grammar, r)
    return tree


def reorder_quantifiers(rule_str):
    matches = re.findall(r'[∃∀]\w', rule_str)
    for match in matches[::-1]:
        rule_str = '%s ' % match + rule_str.replace(match, '', 1)
    return rule_str


def msplit(s):
    for op in op_ls:
        s = s.replace(op, ' %s ' % op)  # add space around
    r = [e.strip() for e in s.split()]
    # remove ' from the string if it contains any: this causes error in nltk cfg parsing
    r = [e.replace('\'', '') for e in r]
    r = [e for e in r if e != '']  # remove blanks

    # deal with symbols with spaces like "dc universe" and turn it to "DcUniverse"
    res = []
    cur_str_ls = []
    for e in r:
        if (len(e) > 1) and sym_reg.match(e):
            cur_str_ls.append(e[0].upper() + e[1:])
        else:
            if len(cur_str_ls) > 0:
                res.extend([''.join(cur_str_ls), e])
            else:
                res.extend([e])
            cur_str_ls = []
    if len(cur_str_ls) > 0:
        res.append(''.join(cur_str_ls))

    # re-generate the FOL string
    make_str_ls = []
    for ind, e in enumerate(r):
        if re.match(r'[⊕∨∧→↔]', e):
            make_str_ls.append(' %s ' % e)  # Add space
        elif re.match(r',', e):
            make_str_ls.append('%s ' % e)
        # a logical variable
        elif (len(e) == 1) and re.match(r'\w', e):
            if ((ind - 1) >= 0) and ((r[ind - 1] == '∃') or (r[ind - 1] == '∀')):
                make_str_ls.append('%s ' % e)  # Attach variable
            else:
                make_str_ls.append(e)
        else:
            make_str_ls.append(e)

    return res, ''.join(make_str_ls)


def make_cfg_str(token_ls):
    """
    NOTE: since nltk does not support reg strs like \w+, we cannot separately recognize VAR, PRED, and CONST.
    Instead, we first allow VAR, PRED, and CONST to be matched with all symbols found in the FOL; once the tree is
    parsered, we then go back and figure out the exact type of each symbols
    """
    sym_ls = list(set([e for e in token_ls if sym_reg.match(e)]))
    sym_str = ' | '.join(["'%s'" % s for s in sym_ls])
    cfg_str = cfg_template + 'VAR -> %s\nPRED -> %s\nCONST -> %s' % (sym_str, sym_str, sym_str)
    return cfg_str


class VecRuleEvaluator:
    dummy_input_str: str = '#DUMMY'
    dummy_distance: int = 10000

    @classmethod
    def default_input_similarity(cls, e1: str, e2: str):
        if e1.startswith(cls.dummy_input_str) or e2.startswith(cls.dummy_input_str):
            return cls.dummy_distance
        return edit_dist(e1, e2)

    @classmethod
    def enumerate_bindings_with_greedy_match(cls, ls1: List[str], ls2: List[str], top_n: int):
        """
            Given two lists of strings ls1 and ls2, yields the ind bindings of ls2 strings that matches the strings in
            ls1. I use greedy match and yields starts from the best to the worst binding until full enumeration or hit
            the top_n bound
        """

        used_inds = []

        def _enum_bindings(ind1: int):
            if ind1 == len(ls1):
                yield deepcopy(used_inds)
                return
            e1 = ls1[ind1]
            match_ls = [
                (ind, cls.default_input_similarity(e1, e2))
                for ind, e2 in enumerate(ls2) if ind not in used_inds
            ]
            match_ls.sort(key=lambda x: x[1])
            for ind, dist in match_ls:
                used_inds.append(ind)
                for inds in _enum_bindings(ind1 + 1):
                    yield inds
                used_inds.pop()

        for cnt, ind_ls in enumerate(_enum_bindings(0)):
            yield ind_ls
            if cnt + 1 == top_n:
                break

    @classmethod
    def find_inputs(cls, root, input_set=None):
        if isinstance(root, str):
            return

        label = root.label()

        # print(label)

        if label == 'L':
            literal_str = ''.join(root.leaves())
            # print(literal_str)
            literal_str = literal_str[1:] if literal_str[0] == '¬' else literal_str
            # print(literal_str)
            if input_set is None:
                input_set = set()
            input_set.add(literal_str)
        else:
            for child in root:
                cls.find_inputs(child, input_set)

    @classmethod
    def gen_input_vecs(cls, num_inputs):
        return np.array(list(product([False, True], repeat=num_inputs)))

    @classmethod
    def from_nltk_tree(cls, root, name2ind_dict, input_vecs):
        assert not isinstance(root, str), 'something wrong with the rule or the algo; you should not parse a leave'

        label = root.label()

        if label == 'S':

            return cls.from_nltk_tree(root[-1], name2ind_dict, input_vecs)

        elif label == 'F':

            # the case F -> L
            if (len(root) == 1) and (root[0].label() == 'L'):
                return cls.from_nltk_tree(root[0], name2ind_dict, input_vecs)

            #     # Handle the case F -> VAR '=' VAR
            # elif len(root) == 3 and isinstance(root[1], str) and root[1] == '=':
            #     var1, var2 = root[0], root[2]
            #     # Assuming var1 and var2 are variables and we need to map them to indices or vectors
            #     if var1 in name2ind_dict and var2 in name2ind_dict:
            #         vec1 = input_vecs[name2ind_dict[var1]]
            #         vec2 = input_vecs[name2ind_dict[var2]]
            #         # Return a boolean value based on the equality comparison
            #         return np.array_equal(vec1, vec2)

            # the case F -> '¬' '(' F ')' | (' F ')'
            elif root[-2].label() == 'F':

                isnegated_rule = isinstance(root[0], str) and (root[0] == '¬')
                res = cls.from_nltk_tree(root[-2], name2ind_dict, input_vecs)

                if isnegated_rule:
                    res = ~res

                return res

            # the case F -> F OP F
            elif root[-2].label() == 'OP':

                p, q = cls.from_nltk_tree(root[0], name2ind_dict, input_vecs), \
                    cls.from_nltk_tree(root[-1], name2ind_dict, input_vecs)

                op = root[1][0]
                if op == '⊕':
                    return np.logical_xor(p, q)
                elif op == '∨':
                    return np.logical_or(p, q)
                elif op == '∧':
                    return np.logical_and(p, q)
                elif op == '→':
                    return np.logical_or(~p, q)
                elif op == '↔':
                    return np.logical_or(np.logical_and(p, q), np.logical_and(~p, ~q))
                else:
                    raise ValueError

        elif label == 'L':

            isnegated_literal = isinstance(root[0], str) and (root[0] == '¬')

            literal_str = ''.join(root.leaves())
            # remove the possible negation at the beginning
            literal_str = literal_str[1:] if isnegated_literal else literal_str

            vec = input_vecs[:, name2ind_dict[literal_str]]

            if isnegated_literal:
                vec = ~vec

            return vec

        else:
            raise ValueError

    @classmethod
    def find_best_LE_score(
            cls,
            true_root,
            pred_root,
            soft_binding: bool,
            greedy_match: bool,
            top_n: int,
            verbose: bool = False
    ):
        """
            Given the groundtruth and the predicted nltk FOL trees, compute the truth tables over all
            literal bindings and returns the best one
        """

        # first we find "inputs" in each tree, i.e. the set of unique literals in a FOL
        true_inputs, pred_inputs = set(), set()
        VecRuleEvaluator.find_inputs(true_root, true_inputs), VecRuleEvaluator.find_inputs(pred_root, pred_inputs)
        true_inputs, pred_inputs = list(true_inputs), list(pred_inputs)
        n_true_inputs, n_pred_inputs = len(true_inputs), len(pred_inputs)
        min_n, max_n = sorted([n_true_inputs, n_pred_inputs])

        # print("True iputs:", true_inputs)

        # print("Predicted inputs:", pred_inputs)

        best_score, best_binded_pred_inputs = 0., None

        # once we found the inputs, then we deal with the case where # inputs in two trees are different
        # either we do soft binding by adding dummy inputs to the shorter ones, or we simply return 0
        if n_true_inputs != n_pred_inputs:
            if soft_binding:
                # extend the shorter inputs to the max number of inputs by adding dummy input names
                ls_to_extend = true_inputs if n_true_inputs < max_n else pred_inputs
                ls_to_extend.extend([f'{cls.dummy_input_str}_{ind}' for ind in range(max_n - min_n)])
            else:
                return best_score, true_inputs, best_binded_pred_inputs

        # at this point, we have two list ofs inputs of the same length and we will find the input binding that yields
        # the best score
        input_vecs = VecRuleEvaluator.gen_input_vecs(len(true_inputs))
        true_name2ind_dict = dict((e, ind) for ind, e in enumerate(true_inputs))
        true_res_vec = VecRuleEvaluator.from_nltk_tree(true_root, true_name2ind_dict, input_vecs)

        ind_binding_enumerator = \
            cls.enumerate_bindings_with_greedy_match(true_inputs, pred_inputs, top_n) if greedy_match \
                else permutations(list(range(max_n)))

        for cnt_ind, binded_pred_inputs_inds in enumerate(ind_binding_enumerator):
            binded_pred_inputs = [pred_inputs[ind] for ind in binded_pred_inputs_inds]
            pred_name2ind_dict = dict((e, ind) for ind, e in enumerate(binded_pred_inputs))
            pred_res_vec = VecRuleEvaluator.from_nltk_tree(pred_root, pred_name2ind_dict, input_vecs)
            score = (pred_res_vec == true_res_vec).mean(dtype=np.float32).item()

            if verbose:
                print('{0}\n{1}\n{2}\n---\n'.format(
                    score, true_inputs, binded_pred_inputs)
                )

            if score > best_score:
                best_score = score
                best_binded_pred_inputs = binded_pred_inputs

            if cnt_ind + 1 >= top_n:
                break

        return best_score, true_inputs, best_binded_pred_inputs


if __name__ == '__main__':
    msplit('∀x (DrinkRegularly(x, coffee) → IsDependentOn(x, caffeine))')