#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from typing import List

from llama_index.indices.service_context import ServiceContext
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.logger.base import LlamaLogger
from llama_index.callbacks.base import CallbackManager

import project_settings as settings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default="what is instruction GPT ?",
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

    service_context: ServiceContext = ServiceContext.from_defaults()
    print(service_context)

    service_context: ServiceContext = ServiceContext(
        llm_predictor=LLMPredictor(),
        prompt_helper=PromptHelper(),
        embed_model=OpenAIEmbedding(api_key=args.openai_api_key),
        node_parser=SimpleNodeParser(),
        llama_logger=LlamaLogger(),
        callback_manager=CallbackManager(handlers=[]),
    )
    print(service_context.embed_model)

    embedding: List[float] = service_context.embed_model.get_text_embedding(args.text)
    print(embedding)
    print(len(embedding))

    embedding: List[float] = service_context.embed_model.get_agg_embedding_from_queries(
        queries=[args.text]
    )
    print(embedding)
    print(len(embedding))
    return


if __name__ == '__main__':
    main()
