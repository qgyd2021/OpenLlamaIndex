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
from llama_index.query_engine.graph_query_engine import ComposableGraphQueryEngine
from llama_index.schema import Document, NodeRelationship
from llama_index.storage.storage_context import StorageContext
from llama_index.response.schema import Response

import project_settings as settings


DEFAULT_NXCLOUD_SUMMARY = """
我们是谁，我们在做什么
牛信云是一家专业提供CPaaS、SaaS等多种接入方式的国际云通信公司。
自成立以来，专注于云通信领域的技术研发，陆续推出了国际短信、国际语音、全球号码、WhatsApp Business、Viber Business等CPaaS服务，以及NXLink、NXCallbot、NXCC等SaaS服务。
目前，牛信云中国区总部坐落在深圳，并在宁波、上海、成都、厦门、香港设立分公司；
国际总部位于新加坡。深度合作客户有阿里巴巴、腾讯、小米、欢聚时代等头部互联网企业，旨在为出海企业提供一站式通信解决方案，帮助更多企业走向世界。
"""


DEFAULT_HUAWEI_SUMMARY = """
华为创立于1987年，是全球领先的ICT（信息与通信）基础设施和智能终端提供商。
我们的20.7万员工遍及170多个国家和地区，为全球30多亿人口提供服务。
我们致力于把数字世界带入每个人、每个家庭、每个组织，构建万物互联的智能世界。
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="牛信是做什么的?", type=str)
    parser.add_argument("--nxcloud_summary", default=DEFAULT_NXCLOUD_SUMMARY, type=str)
    parser.add_argument("--huawei_summary", default=DEFAULT_HUAWEI_SUMMARY, type=str)

    parser.add_argument(
        "--openai_api_key",
        default=settings.environment.get("openai_api_key", default=None, dtype=str),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    storage_context: StorageContext = StorageContext.from_defaults()
    service_context: ServiceContext = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                openai_api_key=args.openai_api_key
            )
        ),
        embed_model=OpenAIEmbedding(api_key=args.openai_api_key),
    )

    # nx index
    document = Document(
        text=args.nxcloud_summary,
        metadata={
            "source": "metadata_source",
        }
    )
    documents: List[Document] = [document]
    nx_index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
    )

    # huawei index
    document = Document(
        text=args.huawei_summary,
        metadata={
            "source": "metadata_source",
        }
    )
    documents: List[Document] = [document]
    hw_index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
    )

    # composable graph
    graph = ComposableGraph.from_indices(
        ListIndex,
        [nx_index, hw_index],
        index_summaries=["牛信云公司的摘要信息(公司简介)", "华为公司的摘要信息(公司简介)"],
        service_context=service_context,
        storage_context=storage_context,
    )

    #
    custom_query_engines = {}
    query_engine = ComposableGraphQueryEngine(
        graph=graph,
        custom_query_engines=custom_query_engines,
        recursive=True
    )
    response: Response = query_engine.query(args.query)
    print(response.response)
    print(len(response.source_nodes))
    for source_node in response.source_nodes:
        print(source_node.node.text)
        print(source_node.score)
        print("-" * 100)

    #
    custom_query_engines = {}
    query_engine = ComposableGraphQueryEngine(
        graph=graph,
        custom_query_engines=custom_query_engines,
        recursive=False
    )
    response: Response = query_engine.query(args.query)
    print(response.response)
    print(len(response.source_nodes))
    for source_node in response.source_nodes:
        print(source_node.node.text)
        print(source_node.score)
        print("-" * 100)

    return


if __name__ == '__main__':
    main()
