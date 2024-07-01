import os
import json
import argparse
import glob
from typing import List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import AutoConfig, AutoModel, AutoTokenizer

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings, HuggingFaceInstructEmbeddings


def find_txt_files(directory: str) -> List[str]:
    return glob.glob(os.path.join(directory, '**/*.txt'), recursive=True)


def get_vectors(homedir: str, embedding: object):
    docs = [TextLoader(f).load() for f in find_txt_files(homedir)]
    docs_list = [item for sublist in docs for item in sublist]

    vectorstore = Chroma.from_documents(
        documents=docs_list,
        collection_name="rag-chroma",
        embedding=embedding,
    )

    vdata = vectorstore.get(0, include=["metadatas", "embeddings"])
    metadata = vdata['metadatas']
    embeddings = vdata['embeddings']

    return metadata, embeddings


def get_embedding_object(embedding_method: str):
    if embedding_method == "GPT4AllEmbeddings":
        return GPT4AllEmbeddings()
    elif embedding_method == "HuggingFaceEmbeddings":
        return HuggingFaceEmbeddings()
    elif embedding_method == "HuggingFaceBgeEmbeddings":
        return HuggingFaceBgeEmbeddings()
    elif embedding_method == "HuggingFaceInstructEmbeddings":
        return HuggingFaceInstructEmbeddings()
    else:
        raise ValueError("Unknown embedding method {embedding_method}")


def extract_embedding_from_classifier(homedir: str):
    # https://arxiv.org/pdf/2310.05791
    def find_txt_files(directory: str) -> List[str]:
        return glob.glob(os.path.join(directory, '**/*.txt'), recursive=True)

    class MultiLabelClassificationHead(nn.Module):
        def __init__(self, num_labels, hidden_size=768):
            super().__init__()
            self.fc = nn.Linear(hidden_size, num_labels)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.fc(x)
            x = self.sigmoid(x)
            return x

    state = torch.load('./classifier_weights.pt',
                       map_location=torch.device('cpu'))
    model_state_dict = {}
    tag_state_dict = {}
    for k, v in state.items():
        if "model." in k:
            name = k[6:]
            model_state_dict[name] = v
        if "tags_classifier." in k:
            name = k[len("tags_classifier."):]
            tag_state_dict[name] = v

    model_config = AutoConfig.from_pretrained(
        "google/bigbird-roberta-base", max_position_embeddings=1024)
    model = AutoModel.from_config(model_config)
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    tag_head = MultiLabelClassificationHead(10)
    model.load_state_dict(model_state_dict)
    tag_head.load_state_dict(tag_state_dict)

    paths_txtfiles = [f for f in find_txt_files(homedir)]
    features = []
    for path_txt in tqdm(paths_txtfiles):
        with torch.no_grad():
            input_ids = tokenizer(open(path_txt).read(), padding=True,
                                  truncation=True, return_tensors='pt', max_length=1024).to("cpu")["input_ids"]
            outputs = model(input_ids=input_ids)
            pooled_output = outputs.pooler_output
            tags_output = tag_head(pooled_output)
        features.append(tags_output[0].cpu().numpy())

    return paths_txtfiles, features


def onehot_category(homedir: str, path_problem_lists: str):
    algorithm_tag = set(['mathematics', 'implementation',
                        'dynamic programming', 'data structures', 'graph theory'])

    problem_numbers = set([int(os.path.splitext(os.path.basename(f))[
                          0]) for f in find_txt_files(homedir)])
    problems = json.load(open(path_problem_lists))

    tags = sorted(set(sum([p["tags"] for p in problems], [])))
    tag2index = {tags[i]: i for i in range(len(tags))}

    paths_txtfiles = []
    features = []
    for prob in problems:
        if prob["problemId"] in problem_numbers:
            feature = np.zeros(len(tags))
            for tag in prob["tags"]:
                if tag not in algorithm_tag:
                    feature[tag2index[tag]] = 1

            paths_txtfiles.append(os.path.join(
                homedir, str(prob['problemId'])+".txt"))
            features.append(feature)

    return paths_txtfiles, features


def save_np(savedir, fpaths, embedding_vectors):
    os.makedirs(savedir, exist_ok=True)
    for fpath, arr_embed_vector in zip(fpaths, embedding_vectors):
        assert len(arr_embed_vector.shape) == 1
        path_save = os.path.join(savedir,
                                 os.path.splitext(os.path.basename(fpath))[0])
        np.save(path_save, arr_embed_vector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--homedir')
    parser.add_argument('--savedir')
    parser.add_argument('--embedding')
    args = parser.parse_args()

    homedir = args.homedir
    savedir = args.savedir
    if args.embedding == "classifier":

        fpaths, embedding_vectors = extract_embedding_from_classifier(homedir)
        save_np(savedir, fpaths, embedding_vectors)

    elif args.embedding == "category":
        path_problem_lists = "problem_lists.json"
        fpaths, embedding_vectors = onehot_category(
            homedir, path_problem_lists)

        save_np(savedir, fpaths, embedding_vectors)

    else:
        embeddings = get_embedding_object(args.embedding)

        metadatas, embedding_vectors = get_vectors(homedir, embeddings)

        os.makedirs(savedir, exist_ok=True)
        for metadata, embed_vector in zip(metadatas, embedding_vectors):
            arr_embed_vector = np.array(embed_vector)
            assert len(arr_embed_vector.shape) == 1
            path_save = os.path.join(savedir,
                                     os.path.splitext(os.path.basename(metadata["source"]))[0])
            np.save(path_save, arr_embed_vector)
