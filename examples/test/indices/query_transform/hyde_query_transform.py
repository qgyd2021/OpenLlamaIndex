#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://gpt-index.readthedocs.io/en/latest/core_modules/query_modules/query_engine/advanced/query_transformations.html#hyde-hypothetical-document-embeddings
"""
import argparse

from langchain.llms.openai import OpenAI
from llama_index.indices.query.query_transform.base import HyDEQueryTransform
from llama_index.indices.query.schema import QueryBundle
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.prompts.default_prompts import DEFAULT_HYDE_PROMPT

import project_settings as settings


def get_args():
    parser = argparse.ArgumentParser()
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

    hyde = HyDEQueryTransform(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                openai_api_key=args.openai_api_key
            )
        ),
        hyde_prompt=DEFAULT_HYDE_PROMPT,
        include_original=True
    )

    result: QueryBundle = hyde.run(args.query)
    print(result)

    return


if __name__ == '__main__':
    main()
