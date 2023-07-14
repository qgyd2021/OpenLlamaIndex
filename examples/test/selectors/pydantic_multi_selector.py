#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

from llama_index.llms.openai import OpenAI
from llama_index.program.openai_program import OpenAIPydanticProgram
from llama_index.selectors.prompts import DEFAULT_MULTI_PYD_SELECT_PROMPT_TMPL
from llama_index.selectors.pydantic_selectors import PydanticMultiSelector
from llama_index.selectors.types import SelectorResult
from llama_index.tools.types import ToolMetadata

import project_settings as settings
from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="牛信是做什么的?", type=str)
    parser.add_argument("--max_outputs", default=2, type=int)

    parser.add_argument(
        "--openai_api_key",
        default=settings.environment.get("openai_api_key", default=None, dtype=str),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    selector: PydanticMultiSelector = PydanticMultiSelector.from_defaults(
        llm=OpenAI(
            model="gpt-3.5-turbo-0613",
            additional_kwargs={
                "api_key": args.openai_api_key
            }
        ),
        prompt_template_str=DEFAULT_MULTI_PYD_SELECT_PROMPT_TMPL,
        max_outputs=args.max_outputs
    )

    choices = [
        ToolMetadata(
            name="NXCloud Summary", description="牛信云公司的摘要信息(公司简介)"
        ),
        ToolMetadata(
            name="Huawei Summary", description="华为公司的摘要信息(公司简介)"
        ),
        ToolMetadata(
            name="Honor Summary", description="荣耀公司的摘要信息(公司简介)"
        ),
        ToolMetadata(
            name="Datetime", description="获取当前时间或时间计算"
        ),
    ]
    result: SelectorResult = selector.select(
        choices=choices,
        query=args.query
    )
    print(result)
    print(type(result))
    return


if __name__ == '__main__':
    main()
