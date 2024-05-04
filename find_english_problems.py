import os
import shutil
import re


def is_written_in_english(text: str, threshold: float = 0.5) -> bool:

    kor_pattern = r'[\u3131-\uD79D]'
    jap_pattern = r'[\u3040-\u30FF\u31F0-\u31FF\u4E00-\u9FFF]'
    rus_pattern = r'[\u0400-\u04FF\u0500-\u052F]'
    polish_pattern = r'[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]'
    thai_pattern = r'[\u0E00-\u0E7F\u0E80-\u0EFF\u0F00-\u0FFF]'
    chinese_pattern = r'[\u4E00-\u9FFF]'

    patterns = [kor_pattern, jap_pattern, rus_pattern, polish_pattern, thai_pattern, chinese_pattern]

    for pattern in patterns:
        match = re.findall(pattern, text)
        if len(match) > 0:
            return False

    return True


def read_text(fpath):
    with open(fpath) as f:
        content = f.read()
    return content


if __name__ == "__main__":

    # dir_problem = "problems"
    # dir_save = "english_problems"
    dir_problem = "problems_desc_only"
    dir_save = "english_problems_desc_only"

    # find english problems
    path_english_problems = []
    for fname in os.listdir(dir_problem):
        fpath = os.path.join(dir_problem, fname)
        text = read_text(fpath)
        if is_written_in_english(text):
            path_english_problems.append(fpath)

    # copy english problems
    os.makedirs(dir_save, exist_ok=True)
    for path_src in path_english_problems:
        shutil.copy(path_src, dir_save)
