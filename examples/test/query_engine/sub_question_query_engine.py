#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://gpt-index.readthedocs.io/en/latest/examples/query_engine/sub_question_query_engine.html
"""
import argparse
from typing import List

from langchain.llms.openai import OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.query_engine import RetrieverQueryEngine, SubQuestionQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.indices.service_context import ServiceContext
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.prompts.default_prompt_selectors import DEFAULT_REFINE_PROMPT_SEL
from llama_index.response.schema import Response
from llama_index.response_synthesizers.compact_and_refine import CompactAndRefine
from llama_index.retrievers import VectorIndexRetriever
from llama_index.schema import Document, NodeWithScore
from llama_index.storage.storage_context import StorageContext
from llama_index.tools import QueryEngineTool, ToolMetadata

import project_settings as settings
from project_settings import project_path


DEFAULT_TEXT = """
我们是谁，我们在做什么
牛信云是一家专业提供CPaaS、SaaS等多种接入方式的国际云通信公司。
自成立以来，专注于云通信领域的技术研发，陆续推出了国际短信、国际语音、全球号码、WhatsApp Business、Viber Business等CPaaS服务，以及NXLink、NXCallbot、NXCC等SaaS服务。
目前，牛信云中国区总部坐落在深圳，并在宁波、上海、成都、厦门、香港设立分公司；
国际总部位于新加坡。深度合作客户有阿里巴巴、腾讯、小米、欢聚时代等头部互联网企业，旨在为出海企业提供一站式通信解决方案，帮助更多企业走向世界。
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_chunk", default=DEFAULT_TEXT, type=str)
    parser.add_argument("--query", default="牛信是做什么的?", type=str)

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

    # query engine
    query_engine = RetrieverQueryEngine(
        retriever=VectorIndexRetriever(
            index=VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                service_context=service_context,
            ),
            similarity_top_k=2,
        ),
        response_synthesizer=CompactAndRefine(
            service_context=service_context,
            text_qa_template=DEFAULT_TEXT_QA_PROMPT,
            refine_template=DEFAULT_REFINE_PROMPT_SEL,
            streaming=False,
        ),
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.1)
        ]
    )

    query_engine_tools = [
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="NXCloud Summary", description="牛信云公司的摘要信息(公司简介)"
            ),
        )
    ]

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    service_context: ServiceContext = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                openai_api_key=args.openai_api_key
            )
        ),
        embed_model=OpenAIEmbedding(api_key=args.openai_api_key),
        callback_manager=callback_manager,
    )
    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        service_context=service_context,
        use_async=False,
    )

    response: Response = query_engine.query(
        "牛信有哪些客户,什么业务"
    )
    print(response.response)
    return


if __name__ == '__main__':
    main()
