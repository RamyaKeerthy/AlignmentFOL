import re
from fol_parser import parse_text_FOL_to_tree
from nltk import Tree

# Helper function to convert CFG parse tree to Penman notation
def parse_tree_to_amr(tree):
    counter = {"count": 0}
    def get_unique_id(base_name):
        counter["count"] += 1
        return f"{base_name}_{counter['count']}"
    """
    Convert an NLTK parse tree (CFG structure) to Penman notation.
    """
    # Recursively traverse the tree and build the Penman structure
    def traverse(node):
        if isinstance(node, Tree):
            label = node.label()
            if label == "S":
                # Check if S has a quantifier and a formula, then combine them
                if len(node) == 2:
                    quantifier_part = traverse(node[0])
                    formula_part = traverse(node[1])
                    # Combine Q and F into one expression with F nested in Q's scope
                    quants = quantifier_part.count("scope")
                    nots = formula_part.count("scope")
                    scopes = quants-nots
                    return f"{quantifier_part[:-scopes]} {formula_part}{')' * scopes}"
                else:
                    return " ".join([traverse(child) for child in node])
            elif label == "Q":
                quantifier = traverse(node[0])
                variable = traverse(node[1])
                unique_quantifier_id = get_unique_id("quant")
                return f"({unique_quantifier_id} / {quantifier}\n   :scope ({variable}"
            elif label == "F":
                if node[0] == "¬":
                    # not_id = get_unique_id("not")
                    return f"(not / NOT :scope {traverse(node[2])})" # "not" is the common variable name
                elif node[0] == "(":
                    return traverse(node[1])  # Handle nested expressions
                elif len(node) == 3:  # Binary operation
                    left = traverse(node[0])
                    op = traverse(node[1])
                    right = traverse(node[2])
                    return f"({op} :left {left} :right {right})" # do not need a variable name
                else:
                    return traverse(node[0])
            elif label == "L":
                if len(node) == 5 and node[0] == "¬":  # Negated predicate or equality
                    pred = traverse(node[1])
                    terms = traverse(node[3]) # node 2 and 4 are parenthesis
                    not_id = get_unique_id("not")
                    return f"({not_id} / NOT :scope ({pred} :args {terms}))"
                elif len(node) == 3:  # Equality
                    left = traverse(node[0])
                    right = traverse(node[2])
                    eq_id = get_unique_id("eq")
                    return f"({eq_id} / EQ :left {left} :right {right})"
                else:
                    pred = traverse(node[0])
                    terms = traverse(node[2])
                    return f"({pred} :args {terms})"
            elif label == "TERMS":
                if len(node) == 3:
                    return f"{traverse(node[0])}, {traverse(node[2])}"
                else:
                    return traverse(node[0])
            elif label == "TERM":
                return traverse(node[0])
            elif label == "OP":
                op_map = {"⊕": "XOR", "∨": "OR", "∧": "AND", "→": "IMPLIES", "↔": "IFF"}
                op_symbol = op_map.get(node[0], node[0])
                unique_op_id = get_unique_id(op_symbol.lower())
                return f"{unique_op_id} / {op_symbol}"
            elif label == "QUANT":
                return node[0]
            elif label == "PRED":
                pred_id = get_unique_id(node[0].lower())
                return f"{pred_id} / {node[0]}"
            elif label == "CONST" or label == "VAR":
                var_id = get_unique_id("var")
                return f"{var_id} / {node[0]}"
        else:
            return node

    return traverse(tree)



if __name__ == '__main__':
    parse_tree = parse_text_FOL_to_tree("∀x (IsEel(x) → IsFish(x))")
    # Convert the parse tree to a graph
    graph = parse_tree_to_amr(parse_tree)
    print(graph)
