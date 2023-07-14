#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://gpt-index.readthedocs.io/en/latest/end_to_end_tutorials/chatbots/building_a_chatbot.html#setting-up-the-tools-langchain-chatbot-agent
"""
import argparse
import os
from typing import Dict, List

from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from llama_index import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.list.base import ListIndex
from llama_index.indices.loading import load_graph_from_storage, load_index_from_storage
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.query_engine.transform_query_engine import TransformQueryEngine

import project_settings as settings
from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
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

    # service_context
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=0,
            max_tokens=512,
            openai_api_key=args.openai_api_key
        )
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        embed_model=OpenAIEmbedding(api_key=args.openai_api_key),
    )
    storage_context = StorageContext.from_defaults()

    # index_dict
    index_dict: Dict[int, VectorStoreIndex] = dict()
    for year in years:
        storage_context = StorageContext.from_defaults(
            persist_dir=os.path.join(args.persist_dir, str(year))
        )
        this_index: VectorStoreIndex = load_index_from_storage(
            storage_context=storage_context,
            service_context=service_context,
        )
        index_dict[year] = this_index

    index_summaries = ["UBER 10-k Filing for {} fiscal year".format(year) for year in years]

    # graph
    graph = ComposableGraph.from_indices(
        ListIndex,
        [index_dict[y] for y in years],
        index_summaries=index_summaries,
        service_context=service_context,
        storage_context=storage_context,
    )
    root_id = graph.root_id

    storage_context.persist(persist_dir=os.path.join(args.persist_dir, "root"))

    # load graph
    graph: ComposableGraph = load_graph_from_storage(
        root_id=root_id,
        service_context=service_context,
        storage_context=storage_context,
    )

    # custom_query_engines
    decompose_transform = DecomposeQueryTransform(
        llm_predictor, verbose=True
    )
    custom_query_engines: Dict[str, BaseQueryEngine] = dict()
    for index in index_dict.values():
        query_engine = index.as_query_engine()
        query_engine = TransformQueryEngine(
            query_engine,
            query_transform=decompose_transform,
            transform_metadata={'index_summary': index.index_struct.summary},
        )
        custom_query_engines[index.index_id] = query_engine

    custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
        response_mode='tree_summarize',
        verbose=True,
    )

    # composable graph query engine
    graph_query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)

    # tool config
    graph_config = IndexToolConfig(
        query_engine=graph_query_engine,
        name="Graph Index",
        description="useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for Uber.",
        tool_kwargs={"return_direct": True}
    )

    index_configs: List[IndexToolConfig] = list()
    for y in range(2019, 2023):
        query_engine = index_dict[y].as_query_engine(
            similarity_top_k=3,
        )
        tool_config = IndexToolConfig(
            query_engine=query_engine,
            name="Vector Index {}".format(y),
            description="useful for when you want to answer queries about the {} SEC 10-K for Uber".format(y),
            tool_kwargs={"return_direct": True}
        )
        index_configs.append(tool_config)

    toolkit = LlamaToolkit(
        index_configs=index_configs + [graph_config],
    )
    memory = ConversationBufferMemory(memory_key="chat_history")

    agent_chain = create_llama_chat_agent(
        toolkit,
        llm=OpenAI(
            temperature=0,
            max_tokens=512,
            openai_api_key=args.openai_api_key,
        ),
        memory=memory,
        verbose=True
    )

    while True:
        text_input = input("User: ")
        if text_input == "Quit":
            break
        response = agent_chain.run(input=text_input)
        print(f'Agent: {response}')

    return


if __name__ == '__main__':
    main()
