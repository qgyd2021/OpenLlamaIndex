#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
from typing import Dict, List

from langchain.llms.openai import OpenAI
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.indices.query.query_transform.prompts import DEFAULT_DECOMPOSE_QUERY_TRANSFORM_TMPL
import project_settings as settings
from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="What were some of the biggest risk factors in 2020 for Uber?",
        type=str
    )
    parser.add_argument(
        "--index_summary",
        default="UBER 10-k Filing for 2020 fiscal year",
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

    query_transform = DecomposeQueryTransform(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                temperature=0,
                max_tokens=512,
                openai_api_key=args.openai_api_key
            )
        )
    )

    result = query_transform.run(
        query_bundle_or_str=args.query,
        metadata={"index_summary": args.index_summary}
    )
    print(result)

    return


if __name__ == '__main__':
    main()
