#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
from typing import Dict, List

from langchain.llms.openai import OpenAI
from llama_index.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.indices.query.query_transform.prompts import DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_PROMPT

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

    query_transform = StepDecomposeQueryTransform(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                temperature=0,
                max_tokens=512,
                openai_api_key=args.openai_api_key
            )
        ),
        step_decompose_query_prompt=DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_PROMPT
    )

    result = query_transform.run(
        query_bundle_or_str=args.query,
        metadata={
            "index_summary": args.index_summary,
            "prev_reasoning": None
        }
    )
    print(result)

    return


if __name__ == '__main__':
    main()
