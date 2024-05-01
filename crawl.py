import os
import random
import requests
import time
import json
import re

from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.request import urlopen


SEARCH_URL = "https://solved.ac/api/v3/search/problem"
PROBLEM_URL = "https://solved.ac/api/v3/problem/show"


def crawl_problem_list(page: int):

    querystring = {"query": "", "page": f"{page}"}

    headers = {"Content-Type": "application/json"}
    response = requests.request(
        "GET", SEARCH_URL, headers=headers, params=querystring)

    summarized_problem_lists = []
    problem_list = json.loads(response.text).get("items")
    for prob_desc in problem_list:
        if "problemId" not in prob_desc:
            continue

        if "titleKo" not in prob_desc:
            continue

        # retrieve tags
        retrieved_tags = []
        tags = prob_desc.get("tags", [])
        for tag in tags:
            displayNames = tag.get("displayNames", [])
            for display_name in displayNames:
                if display_name.get("language", "") == "en" and "name" in display_name:
                    retrieved_tags.append(display_name.get("name"))

        summarized_problem_lists.append(
            {"problemId": prob_desc["problemId"], "titleKo": prob_desc["titleKo"], "tags": retrieved_tags})

    return summarized_problem_lists


def crawl_problem_content(problem_id: int):

    user_agents_list = [
        'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.83 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.11 (KHTML, like Gecko) Ubuntu/11.10 Chromium/17.0.963.65 Chrome/17.0.963.65 Safari/535.11',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.11 (KHTML, like Gecko) Ubuntu/11.04 Chromium/17.0.963.65 Chrome/17.0.963.65 Safari/535.11',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.11 (KHTML, like Gecko) Ubuntu/10.10 Chromium/17.0.963.65 Chrome/17.0.963.65 Safari/535.11',
        'Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.11 (KHTML, like Gecko) Ubuntu/11.10 Chromium/17.0.963.65 Chrome/17.0.963.65 Safari/535.11',
        'Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.65 Safari/535.11',
        'Mozilla/5.0 (X11; FreeBSD amd64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.65 Safari/535.11',
        'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.65 Safari/535.11',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_2) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.65 Safari/535.11',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.65 Safari/535.11',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_4) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.65 Safari/535.11']

    HEADERS = {
        'User-Agent': random.choice(user_agents_list)}

    response = requests.get(
        f"https://www.acmicpc.net/problem/{problem_id}", headers=HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')
    problem_desc = soup.select('#problem_description')
    return "".join([e.text for e in problem_desc])


if __name__ == "__main__":

    path_problem_lists = "problem_lists.json"
    second_wait_api = 1
    second_wait_webpage = 5

    # # save problem lists to single json file
    # all_problem_lists = []
    # for page in tqdm(range(1, 603)):
    #     problem_lists = crawl_problem_list(page)
    #     all_problem_lists.extend(problem_lists)
    #     time.sleep(second_wait_api)
    # with open(path_problem_lists, "w", encoding='utf8') as f:
    #     json.dump(all_problem_lists, f, indent="\t", ensure_ascii=False)

    # save problems to files
    dir_save = "problems"
    os.makedirs(dir_save, exist_ok=True)
    with open(path_problem_lists, "r") as f:
        all_problem_lists = json.load(f)
    for problem_desc in tqdm(all_problem_lists):
        title = re.sub(r'[^\w\s]', '', problem_desc["titleKo"])
        problem_id = problem_desc["problemId"]
        path_save = os.path.join(dir_save, f"{title}_{problem_id}.txt")
        if os.path.exists(path_save):
            continue
        
        try:
            problem_content = crawl_problem_content(problem_id)
            with open(path_save, "w") as f:
                f.write(problem_content)
        except:
            pass        
        time.sleep(second_wait_webpage)
