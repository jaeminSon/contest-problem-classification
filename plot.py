import os
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.manifold import TSNE


LEVEL = ["Unrated /  Not Ratable", "Bronze V", "Bronze IV", "Bronze III", "Bronze II", "Bronze I",
         "Silver V", "Silver IV", "Silver III", "Silver II",  "Silver I",
         "Gold V",  "Gold IV",  "Gold III",  "Gold II",  "Gold I",
         "Platinum V",  "Platinum IV",  "Platinum III",  "Platinum II",  "Platinum I",
         "Diamond V",  "Diamond IV",  "Diamond III",  "Diamond II",  "Diamond I",
         "Ruby V",  "Ruby IV",  "Ruby III",  "Ruby II",  "Ruby I"]

ALGORITHM_TAGS = ['mathematics', 'implementation',
                  'dynamic programming', 'data structures', 'graph theory']


def read_problem_list():
    dir_english_problems = "english_problems_desc_only_deduplicate"
    with open("problem_lists.json", "r") as f:
        all_content = json.load(f)
    return [e for e in all_content if os.path.exists(os.path.join(dir_english_problems, f"{e['problemId']}.txt"))]


def draw_level(problem_list):
    counter = Counter([e["level"] for e in problem_list])
    levels = sorted(counter.keys())
    plt.close()
    plt.figure(figsize=(16, 16))
    plt.pie([counter[i] for i in levels],
            labels=[LEVEL[i] for i in levels],
            autopct='%1.1f%%',
            colors=plt.cm.Greys([1.*i/len(levels) for i in range(len(levels))]))
    plt.axis('equal')
    plt.title('Difficulty English Problems')
    plt.savefig("difficulty_english_problems_desc_only_deduplicate.png")


def draw_algo_tags(problem_list):
    counter = Counter(sum([e["tags"] for e in problem_list], []))
    sorted_n_appreance = sorted(counter.values(), reverse=True)
    plt.close()
    plt.plot(sorted_n_appreance)
    plt.title('Algorithm Tag Counts for English Problems')
    plt.xlabel("Rank of algorithms by appearance")
    plt.ylabel("# Appearance")
    plt.savefig(
        "all_algorithm_tags_dist_english_problems_desc_only_deduplicate.png")

    n_top = 30
    algorithms = sorted([e for e in counter
                        if counter[e] >= sorted_n_appreance[n_top-1]],
                        key=lambda x: counter[x], reverse=True)
    plt.close()
    plt.scatter(range(len(algorithms)), [counter[e]
                for e in algorithms], marker="x")
    plt.xticks(range(len(algorithms)), algorithms, rotation=90)
    plt.ylabel("# Appearance")
    plt.title(f'Top {n_top} Algorithm Tags for English Problems')
    plt.tight_layout()
    plt.savefig(
        f"top{n_top}_algorithm_tags_dist_english_problems_desc_only_deduplicate.png", bbox_inches='tight')


def get_label(problem_ids, problemid2algo, target):
    return [1 if target in problemid2algo[int(pid)] else 0 for pid in problem_ids]


def draw_tsne(embedding_method: str, problemid2algo: List[Dict]):

    home_embedding_vectors = Path(embedding_method)
    problem_ids = [os.path.splitext(path_embedding.name)[0]
                   for path_embedding in home_embedding_vectors.iterdir()]
    list_embedding = [np.load(path_embedding)
                      for path_embedding in home_embedding_vectors.iterdir()]
    embeddings = np.stack(list_embedding, axis=0)

    tsne = TSNE(n_components=2)
    tsne_vectors = tsne.fit_transform(embeddings)

    for target_algorithm in ALGORITHM_TAGS:
        colormap = colors.ListedColormap(['gray', 'r'])
        scatter = plt.scatter(tsne_vectors[:, 0],
                              tsne_vectors[:, 1],
                              s=3, c=get_label(problem_ids, problemid2algo, target_algorithm), cmap=colormap)
        plt.legend(handles=scatter.legend_elements()[
                   0], labels=["None", target_algorithm])
        plt.savefig(f"{embedding_method}-{target_algorithm}.png")
        plt.close()


def get_problemid2algo(problem_list):
    problemid2algo = defaultdict(dict)
    for prob in problem_list:
        for algo in prob["tags"]:
            problemid2algo[prob["problemId"]][algo] = True
    return problemid2algo


if __name__ == "__main__":

    problem_list = read_problem_list()

    # draw_level(problem_list)

    # draw_algo_tags(problem_list)

    problemid2algo = get_problemid2algo(problem_list)
    draw_tsne("GPT4AllEmbeddings", problemid2algo)
    draw_tsne("HuggingFaceEmbeddings", problemid2algo)
    draw_tsne("HuggingFaceBgeEmbeddings", problemid2algo)
    draw_tsne("HuggingFaceInstructEmbeddings", problemid2algo)
