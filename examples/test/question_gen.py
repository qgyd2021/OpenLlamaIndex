#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

from langchain.llms.openai import OpenAI
from llama_index.indices.query.schema import QueryBundle
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.prompts.base import Prompt
from llama_index.prompts.prompt_type import PromptType
from llama_index.question_gen.llm_generators import LLMQuestionGenerator
from llama_index.question_gen.output_parser import SubQuestionOutputParser
from llama_index.question_gen.prompts import DEFAULT_SUB_QUESTION_PROMPT_TMPL
from llama_index.tools import ToolMetadata

import project_settings as settings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="牛信有哪些客户,什么业务?", type=str)
    parser.add_argument(
        "--openai_api_key",
        default=settings.environment.get("openai_api_key", default=None, dtype=str),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    question_gen = LLMQuestionGenerator(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                openai_api_key=args.openai_api_key
            )
        ),
        prompt=Prompt(
            template=DEFAULT_SUB_QUESTION_PROMPT_TMPL,
            output_parser=SubQuestionOutputParser(),
            prompt_type=PromptType.SUB_QUESTION,
        )
    )

    metadata = ToolMetadata(
        name="NXCloud Summary", description="牛信云公司的摘要信息(公司简介)"
    )
    query = QueryBundle(
        query_str=args.query
    )

    sub_questions = question_gen.generate(
        tools=[metadata],
        query=query
    )
    print(sub_questions)

    return


if __name__ == '__main__':
    main()
