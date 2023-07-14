#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

from llama_index.indices.service_context import ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.selectors.llm_selectors import LLMSingleSelector
from llama_index.selectors.types import SelectorResult
from llama_index.tools.types import ToolMetadata

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

    service_context: ServiceContext = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                additional_kwargs={
                    "api_key": args.openai_api_key
                }
            )
        ),
    )

    selector: LLMSingleSelector = LLMSingleSelector.from_defaults(
        service_context=service_context,
    )

    choices = [
        ToolMetadata(
            name="NXCloud Summary", description="牛信云公司的摘要信息(公司简介)"
        ),
        ToolMetadata(
            name="Huawei Summary", description="华为公司的摘要信息(公司简介)"
        ),
    ]
    result: SelectorResult = selector.select(
        choices=choices,
        query=args.query
    )
    print(result)
    return


if __name__ == '__main__':
    main()
