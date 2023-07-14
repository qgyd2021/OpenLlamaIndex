#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://gpt-index.readthedocs.io/en/latest/end_to_end_tutorials/chatbots/building_a_chatbot.html
"""
import argparse
import os
from pathlib import Path
from typing import Dict, List

from langchain.llms.openai import OpenAI
from llama_index import download_loader, VectorStoreIndex, ServiceContext, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.schema import Document

import project_settings as settings
from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loader_class", default="UnstructuredReader", type=str)
    parser.add_argument(
        "--data_dir",
        default=(project_path / "examples/tutorial/sec_10k_filings_chatbot/data_dir/UBER").as_posix(),
        type=str
    )

    parser.add_argument("--chunk_size", default=512, type=int)

    parser.add_argument(
        "--persist_dir",
        default=(project_path / "cache/persist_dir").as_posix(),
        type=str
    )
    parser.add_argument(
        "--openai_api_key",
        default=settings.environment.get("openai_api_key", default=None, dtype=str),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    years = [2022, 2021, 2020, 2019]

    loader_class = download_loader(args.loader_class, refresh_cache=True)
    loader = loader_class()

    doc_dict: Dict[int, List[Document]] = dict()
    all_docs = list()
    for year in years:
        year_docs: List[Document] = loader.load_data(
            file=Path("{}/UBER_{}.html".format(args.data_dir, year)),
            split_documents=False
        )
        for d in year_docs:
            d.metadata = {"year": year}

        doc_dict[year] = year_docs
        all_docs.extend(year_docs)

    service_context = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                temperature=0,
                max_tokens=512,
                openai_api_key=args.openai_api_key
            )
        ),
        embed_model=OpenAIEmbedding(api_key=args.openai_api_key),
        chunk_size=args.chunk_size,
    )

    index_dict: Dict[int, VectorStoreIndex] = dict()
    for year in years:
        storage_context = StorageContext.from_defaults()
        this_index = VectorStoreIndex.from_documents(
            doc_dict[year],
            storage_context=storage_context,
            service_context=service_context,
        )
        index_dict[year] = this_index
        storage_context.persist(persist_dir=os.path.join(args.persist_dir, str(year)))

    return


if __name__ == '__main__':
    main()
