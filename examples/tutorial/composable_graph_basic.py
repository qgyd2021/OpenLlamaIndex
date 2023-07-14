#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
参考链接:
https://gpt-index.readthedocs.io/en/latest/core_modules/data_modules/index/composability.html
https://gpt-index.readthedocs.io/en/latest/examples/composable_indices/ComposableIndices-Prior.html
"""
import argparse
import os

from langchain.llms.openai import OpenAI
from llama_index import (
    VectorStoreIndex,
    EmptyIndex,
    TreeIndex,
    ListIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.composability import ComposableGraph
from llama_index.llm_predictor.base import LLMPredictor
import requests

from project_settings import project_path
import project_settings as settings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="What is instructGPT ?", type=str)

    parser.add_argument(
        "--paul_graham_essay_url",
        default="https://raw.githubusercontent.com/jerryjliu/llama_index/main/examples/paul_graham_essay/data/paul_graham_essay.txt",
        type=str
    )
    parser.add_argument(
        "--input_dir",
        default=(project_path / "cache/example_txt").as_posix(),
        type=str
    )
    parser.add_argument("--persist_dir", default=None, type=str)

    parser.add_argument(
        "--openai_api_key",
        default=settings.environment.get("openai_api_key", default=None, dtype=str),
        type=str
    )
    args = parser.parse_args()
    return args


def download_txt(url: str = None, dest_path: str = "."):

    if url is not None:
        _, fn = os.path.split(url)
        filename = os.path.join(dest_path, fn)
        if os.path.exists(filename):
            return filename

        resp = requests.get(url)
        os.makedirs(dest_path, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(resp.text)

        return filename
    return None


def main():
    args = get_args()

    filename: str = download_txt(args.paul_graham_essay_url, args.input_dir)
    if filename is None:
        raise AssertionError

    simple_directory_reader = SimpleDirectoryReader(input_files=[
        filename
    ])
    documents = simple_directory_reader.load_data()

    # configure
    service_context: ServiceContext = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                openai_api_key=args.openai_api_key
            )
        ),
        embed_model=OpenAIEmbedding(api_key=args.openai_api_key),
        chunk_size=512
    )
    storage_context = StorageContext.from_defaults()

    # build index
    essay_index = VectorStoreIndex.from_documents(
        documents, service_context=service_context, storage_context=storage_context
    )
    query_engine = essay_index.as_query_engine(
        similarity_top_k=3,
        response_mode="tree_summarize",
    )
    response = query_engine.query(
        "Tell me about what Sam Altman did during his time in YC",
    )
    print(str(response))

    empty_index = EmptyIndex(
        service_context=service_context, storage_context=storage_context
    )
    query_engine = empty_index.as_query_engine(
        response_mode="generation"
    )
    response = query_engine.query(
        "Tell me about what Sam Altman did during his time in YC",
    )
    print(str(response))

    essay_index_summary = "This document describes Paul Graham's life, from early adulthood to the present day."
    empty_index_summary = "This can be used for general knowledge purposes."

    graph = ComposableGraph.from_indices(
        ListIndex,
        [essay_index, empty_index],
        index_summaries=[essay_index_summary, empty_index_summary],
        service_context=service_context,
        storage_context=storage_context,
    )
    return


if __name__ == '__main__':
    main()
