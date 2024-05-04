import os
import shutil


def read_text(fpath):
    with open(fpath) as f:
        content = f.read()
    return content


if __name__ == "__main__":

    dir_src = "english_problems_desc_only"
    dir_save = "english_problems_desc_only_deduplicate"

    # assume duplicate problems occur sequentially
    selected = []
    fpaths = sorted([os.path.join(dir_src, fname) for fname in os.listdir(dir_src)])
    for i in range(len(fpaths)-1):
        curr_text = read_text(fpaths[i])
        next_text = read_text(fpaths[i+1])
        if curr_text == next_text:
            print(f"skip {fpaths[i]}...")
        else:
            selected.append(fpaths[i])

    # copy selected problems
    os.makedirs(dir_save, exist_ok=True)
    for path_src in selected:
        shutil.copy(path_src, dir_save)
