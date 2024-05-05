import os
import json
import matplotlib.pyplot as plt
from collections import Counter

if __name__ == "__main__":

    labels = ["Unrated /  Not Ratable", "Bronze V", "Bronze IV", "Bronze III", "Bronze II", "Bronze I",
              "Silver V", "Silver IV", "Silver III", "Silver II",  "Silver I",
              "Gold V",  "Gold IV",  "Gold III",  "Gold II",  "Gold I",
              "Platinum V",  "Platinum IV",  "Platinum III",  "Platinum II",  "Platinum I",
              "Diamond V",  "Diamond IV",  "Diamond III",  "Diamond II",  "Diamond I",
              "Ruby V",  "Ruby IV",  "Ruby III",  "Ruby II",  "Ruby I"]

    dir_english_problems = "english_problems_desc_only_deduplicate"
    with open("problem_lists.json", "r") as f:
        all_content = json.load(f)
    content = [e for e in all_content if os.path.exists(os.path.join(dir_english_problems, f"{e['problemId']}.txt"))]

    counter = Counter([e["level"] for e in content])

    levels = sorted(counter.keys())
    plt.figure(figsize=(8, 8))
    plt.pie([counter[i] for i in levels],
            labels=[labels[i] for i in levels],
            autopct='%1.1f%%',
            colors=plt.cm.Greys([1.*i/len(levels) for i in range(len(levels))]))
    plt.axis('equal')
    plt.title('Difficulty English Problems')
    plt.savefig("difficulty_english_problems_desc_only_deduplicate.png")
