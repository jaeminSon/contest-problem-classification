import os
import argparse
import glob
from typing import List

import numpy as np
from tqdm import tqdm

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings, HuggingFaceInstructEmbeddings


def get_vectors(homedir: str, embedding: object):

    def find_txt_files(directory: str) -> List[str]:
        return glob.glob(os.path.join(directory, '**/*.txt'), recursive=True)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--homedir')
    parser.add_argument('--savedir')
    parser.add_argument('--embedding')
    args = parser.parse_args()

    homedir = args.homedir
    embeddings = get_embedding_object(args.embedding)
    savedir = args.savedir

    metadatas, embedding_vectors = get_vectors(homedir, embeddings)

    os.makedirs(savedir, exist_ok=True)
    for metadata, embed_vector in zip(metadatas, embedding_vectors):
        arr_embed_vector = np.array(embed_vector)
        assert len(arr_embed_vector.shape) == 1
        path_save = os.path.join(savedir,
                                 os.path.splitext(os.path.basename(metadata["source"]))[0])
        np.save(path_save, arr_embed_vector)
