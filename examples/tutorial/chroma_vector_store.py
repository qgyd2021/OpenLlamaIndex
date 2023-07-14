#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
参考链接:
https://gpt-index.readthedocs.io/en/latest/getting_started/customization.html
"""
import argparse
import logging
import sys
from typing import List

import chromadb
from langchain.llms.openai import OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.service_context import ServiceContext
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.schema import Document
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore

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
    parser.add_argument("--chunk_size", default=1000, type=int)
    parser.add_argument("--chunk_overlap", default=200, type=int)
    parser.add_argument("--persist_dir", default=None, type=str)

    parser.add_argument("--similarity_top_k", default=5, type=int)
    parser.add_argument("--streaming", action="store_true")

    parser.add_argument(
        "--openai_api_key",
        default=settings.environment.get("openai_api_key", default=None, dtype=str),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    directory_reader = SimpleDirectoryReader(input_dir=args.input_dir)
    documents: List[Document] = directory_reader.load_data()

    # service_context
    service_context: ServiceContext = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                openai_api_key=args.openai_api_key
            )
        ),
        embed_model=OpenAIEmbedding(api_key=args.openai_api_key),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # storage_context
    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.create_collection("quickstart")
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )
    storage_context: StorageContext = StorageContext.from_defaults(vector_store=vector_store)

    # index
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
    )

    query_engine: RetrieverQueryEngine = index.as_query_engine(
        similarity_top_k=args.similarity_top_k,
        response_mode="tree_summarize",
        streaming=args.streaming
    )
    response = query_engine.query(args.query)
    if args.streaming:
        print(response.print_response_stream())
    else:
        print(response)
    return


if __name__ == '__main__':
    main()
