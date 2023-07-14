#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from typing import List

from langchain.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.composability.graph import ComposableGraph
from llama_index.indices.empty.base import EmptyIndex
from llama_index.indices.list.base import ListIndex
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.schema import Document
from llama_index.storage.storage_context import StorageContext

import project_settings as settings


DEFAULT_TEXT = """
我们是谁，我们在做什么
牛信云是一家专业提供CPaaS、SaaS等多种接入方式的国际云通信公司。
自成立以来，专注于云通信领域的技术研发，陆续推出了国际短信、国际语音、全球号码、WhatsApp Business、Viber Business等CPaaS服务，以及NXLink、NXCallbot、NXCC等SaaS服务。
目前，牛信云中国区总部坐落在深圳，并在宁波、上海、成都、厦门、香港设立分公司；
国际总部位于新加坡。深度合作客户有阿里巴巴、腾讯、小米、欢聚时代等头部互联网企业，旨在为出海企业提供一站式通信解决方案，帮助更多企业走向世界。
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="牛信是做什么的?", type=str)
    parser.add_argument("--text_chunk", default=DEFAULT_TEXT, type=str)

    parser.add_argument(
        "--openai_api_key",
        default=settings.environment.get("openai_api_key", default=None, dtype=str),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    document = Document(
        text=args.text_chunk,
        metadata={
            "source": "metadata_source",
        }
    )
    documents: List[Document] = [document]

    storage_context: StorageContext = StorageContext.from_defaults()
    service_context: ServiceContext = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                openai_api_key=args.openai_api_key
            )
        ),
        embed_model=OpenAIEmbedding(api_key=args.openai_api_key),
    )

    essay_index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
    )
    essay_index_summary = "essay_index_summary"

    empty_index = EmptyIndex()
    empty_index_summary = "empty_index_summary"

    graph = ComposableGraph.from_indices(
        ListIndex,
        [essay_index, empty_index],
        index_summaries=[essay_index_summary, empty_index_summary],
        service_context=service_context,
        storage_context=storage_context,
    )
    print(graph.root_id)
    print(graph.all_indices)
    print(graph.root_index)
    print(graph.index_struct)
    print(graph.service_context)

    return


if __name__ == '__main__':
    main()
