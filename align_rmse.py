import numpy as np
from collections import defaultdict
import json
from run_eval_sample import eval_samples

# Pairwise comparison to extract final human ranking
def rank_from_comparisons(*lists):
    elements = set(lists[0])
    win_count = defaultdict(int)

    # Perform pairwise comparisons
    for a in elements:
        for b in elements:
            if a == b:
                continue
            # Count how many times 'a' ranks better than 'b' across the lists
            a_wins = sum(lst.index(a) < lst.index(b) for lst in lists)
            # If 'a' wins over 'b' in more than half the lists, increment the win count for 'a'
            if a_wins > len(lists) // 2:
                win_count[a] += 1

    # Sort the elements by win count in descending order
    final_ranking = sorted(elements, key=lambda x: win_count[x], reverse=True)
    return final_ranking


# Function to calculate RMSE between two lists
def calculate_rmse(judge1, judge2):
    differences = np.subtract(judge1, judge2)  # Element-wise difference
    squared_diff = np.square(differences)      # Element-wise square of the differences
    mean_squared_diff = np.mean(squared_diff)  # Mean of the squared differences
    rmse = np.sqrt(mean_squared_diff)          # Square root of the mean
    return rmse


if __name__ == '__main__':
    with open('human_rank.json', 'r') as file:
        human_evals = json.load(file)
    final_human_ranks = []
    for row in human_evals:
        final_human_ranks.append(row['human1'], row['human2'], row['human3'])

    rankbleu, rankle, rankbert, rankrouge, rankmeteor, ranksmatch = eval_samples("gpt_fol_samples.json", save_file=False)
    print(f"RMSE bleu: {calculate_rmse(final_human_ranks, rankbleu)}")
    print(f"RMSE LE: {calculate_rmse(final_human_ranks, rankle)}")
    print(f"RMSE BERT: {calculate_rmse(final_human_ranks, rankbert)}")
    print(f"RMSE ROUGE: {calculate_rmse(final_human_ranks, rankrouge)}")
    print(f"RMSE METEOR: {calculate_rmse(final_human_ranks, rankmeteor)}")
    print(f"RMSE SMATCH: {calculate_rmse(final_human_ranks, ranksmatch)}")
