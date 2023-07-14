#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
参考链接:
https://gpt-index.readthedocs.io/en/latest/getting_started/starter_example.html
"""
import argparse
import logging
import sys
from typing import List

from langchain.llms.openai import OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.indices.service_context import ServiceContext
from llama_index.schema import Document
from llama_index.storage.storage_context import StorageContext
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine

from project_settings import project_path
import project_settings as settings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="What is instructGPT ?", type=str)

    parser.add_argument(
        "--input_dir",
        default=(project_path / "cache/arxiv_pdf").as_posix(),
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


def main():
    args = get_args()

    # logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    directory_reader = SimpleDirectoryReader(input_dir=args.input_dir)
    documents: List[Document] = directory_reader.load_data()
    # print(documents)

    storage_context: StorageContext = StorageContext.from_defaults(persist_dir=args.persist_dir)
    service_context: ServiceContext = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                openai_api_key=args.openai_api_key
            )
        ),
        embed_model=OpenAIEmbedding(api_key=args.openai_api_key),
    )

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
    )

    query_engine: RetrieverQueryEngine = index.as_query_engine()
    response = query_engine.query(args.query)
    print(response)
    return


if __name__ == '__main__':
    main()
