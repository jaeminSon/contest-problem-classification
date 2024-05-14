import os
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

        os.makedirs(savedir, exist_ok=True)
        for fpath, arr_embed_vector in zip(fpaths, embedding_vectors):
            assert len(arr_embed_vector.shape) == 1
            path_save = os.path.join(savedir,
                                     os.path.splitext(os.path.basename(fpath))[0])
            np.save(path_save, arr_embed_vector)

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
