#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://gpt-index.readthedocs.io/en/latest/examples/query_engine/RouterQueryEngine.html
"""
import argparse
from typing import List

# from langchain.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.composability.graph import ComposableGraph
from llama_index.indices.empty.base import EmptyIndex
from llama_index.indices.list.base import ListIndex
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.indices.list.retrievers import ListIndexRetriever
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.prompts.default_prompt_selectors import DEFAULT_REFINE_PROMPT_SEL
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.selectors.pydantic_selectors import PydanticSingleSelector
from llama_index.schema import Document, NodeRelationship
from llama_index.storage.storage_context import StorageContext
from llama_index.readers.file.base import SimpleDirectoryReader
from llama_index.response.schema import Response
from llama_index.response_synthesizers.compact_and_refine import CompactAndRefine
from llama_index.tools.query_engine import QueryEngineTool

import project_settings as settings
from project_settings import project_path


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
        "--input_file",
        default=(project_path / "cache/example_txt/paul_graham_essay.txt").as_posix(),
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

    storage_context: StorageContext = StorageContext.from_defaults()
    service_context: ServiceContext = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                model="text-davinci-003",
                additional_kwargs={
                    "api_key": args.openai_api_key
                }
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
    nx_index: ListIndex = ListIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
    )
    nx_retriever = ListIndexRetriever(
        index=nx_index,
    )
    nx_query_engine: RetrieverQueryEngine = RetrieverQueryEngine.from_args(
        retriever=nx_retriever,
        service_context=service_context,
        response_mode="tree_summarize",
    )
    nx_tool = QueryEngineTool.from_defaults(
        query_engine=nx_query_engine,
        description="牛信云公司的摘要信息(公司简介)",
    )

    # huawei index
    document = Document(
        text=args.huawei_summary,
        metadata={
            "source": "metadata_source",
        }
    )
    documents: List[Document] = [document]
    hw_index: ListIndex = ListIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
    )
    hw_retriever = ListIndexRetriever(
        index=hw_index,
    )
    hw_query_engine: RetrieverQueryEngine = RetrieverQueryEngine.from_args(
        retriever=hw_retriever,
        service_context=service_context,
        response_mode="tree_summarize",
    )
    hw_tool = QueryEngineTool.from_defaults(
        query_engine=hw_query_engine,
        description="华为公司的摘要信息(公司简介)",
    )

    # query_engine
    query_engine = RouterQueryEngine(
        selector=PydanticSingleSelector.from_defaults(
            llm=OpenAI(
                model="gpt-3.5-turbo-0613",
                additional_kwargs={
                    "api_key": args.openai_api_key
                }
            )
        ),
        query_engine_tools=[
            nx_tool,
            hw_tool,
        ],
        service_context=service_context
    )

    response: Response = query_engine.query("牛信云是做什么的?")
    print(response.response)
    return


if __name__ == '__main__':
    main()
