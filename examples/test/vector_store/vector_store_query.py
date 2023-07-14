#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from typing import List

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.vector_stores.types import VectorStoreQuery
from llama_index.embeddings.langchain import LangchainEmbedding

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="sentence-transformers/all-mpnet-base-v2",
        type=str
    )
    parser.add_argument(
        "--pretrained_models_dir",
        default=(project_path / "pretrained_models").as_posix(),
        type=str
    )
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--normalize_embeddings", action="store_true")

    parser.add_argument("--query", default="what the date today ?", type=str)

    parser.add_argument("--similarity_top_k", default=3, type=int)
    parser.add_argument(
        "--vector_store_query_mode",
        default=3,
        choices=["default", "sparse", "hybrid", "text_search",
                 "svm", "logistic_regression", "linear_regression", "mmr"],
        type=str,
    )
    parser.add_argument("--alpha", default=0.5, type=float)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    hf = HuggingFaceEmbeddings(
        model_name=args.model_name,
        cache_folder=args.pretrained_models_dir,
        model_kwargs={
            "device": args.device,
        },
        encode_kwargs={
            "normalize_embeddings": args.normalize_embeddings
        },
    )
    embed_model = LangchainEmbedding(
        langchain_embedding=hf
    )

    embedding = embed_model.get_text_embedding(args.query)

    query = VectorStoreQuery(
        query_embedding=embedding,
        similarity_top_k=args.similarity_top_k,
        query_str=args.query,
        mode=args.vector_store_query_mode,
        alpha=args.alpha,
    )
    print(query)
    return


if __name__ == '__main__':
    main()
