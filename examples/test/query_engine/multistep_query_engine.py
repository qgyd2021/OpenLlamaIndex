#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://gpt-index.readthedocs.io/en/latest/examples/query_transformations/SimpleIndexDemo-multistep.html
"""
import argparse
from typing import List

from langchain.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.prompts.default_prompt_selectors import DEFAULT_REFINE_PROMPT_SEL
from llama_index.query_engine.multistep_query_engine import MultiStepQueryEngine
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.readers.file.base import SimpleDirectoryReader
from llama_index.response.schema import Response
from llama_index.response_synthesizers.compact_and_refine import CompactAndRefine
from llama_index.schema import Document
from llama_index.storage.storage_context import StorageContext

import project_settings as settings
from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="牛信是做什么的?", type=str)

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

    directory_reader = SimpleDirectoryReader(input_files=[args.input_file])
    documents: List[Document] = directory_reader.load_data()

    # index
    storage_context: StorageContext = StorageContext.from_defaults(
        persist_dir=args.persist_dir
    )
    service_context: ServiceContext = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                openai_api_key=args.openai_api_key
            )
        ),
        embed_model=OpenAIEmbedding(
            api_key=args.openai_api_key
        ),
    )
    index: VectorStoreIndex = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
    )
    # query_engine
    query_engine = RetrieverQueryEngine(
        retriever=VectorIndexRetriever(
            index=index,
            similarity_top_k=2
        ),
        response_synthesizer=CompactAndRefine(
            service_context=service_context,
            text_qa_template=DEFAULT_TEXT_QA_PROMPT,
            refine_template=DEFAULT_REFINE_PROMPT_SEL,
            streaming=False,
        )
    )

    # query_transform
    step_decompose_transform = StepDecomposeQueryTransform(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                openai_api_key=args.openai_api_key
            )
        ),
        verbose=True
    )

    query_engine = MultiStepQueryEngine(
        query_engine=query_engine,
        query_transform=step_decompose_transform,
        response_synthesizer=CompactAndRefine(
            service_context=service_context,
            text_qa_template=DEFAULT_TEXT_QA_PROMPT,
            refine_template=DEFAULT_REFINE_PROMPT_SEL,
            streaming=False,
        ),
        index_summary="Used to answer questions about the author",
    )

    response: Response = query_engine.query(
        "Who was in the first batch of the accelerator program the author started?",
    )
    print(response.response)
    # print(response.source_nodes)
    return


if __name__ == '__main__':
    main()
