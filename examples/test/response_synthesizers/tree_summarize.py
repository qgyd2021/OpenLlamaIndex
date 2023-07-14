#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from typing import List

from langchain.llms.openai import OpenAI
from llama_index.indices.service_context import ServiceContext
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.prompts.default_prompts import DEFAULT_SIMPLE_INPUT_PROMPT, DEFAULT_TEXT_QA_PROMPT
from llama_index.prompts.default_prompt_selectors import DEFAULT_REFINE_PROMPT_SEL
from llama_index.response_synthesizers.tree_summarize import TreeSummarize

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
    parser.add_argument("--query", default="该文档的摘要是什么?", type=str)
    parser.add_argument("--text_chunk", default=DEFAULT_TEXT, type=str)

    parser.add_argument(
        "--openai_api_key",
        default=settings.environment.get("openai_api_key", default=None, dtype=str),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    service_context = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                openai_api_key=args.openai_api_key
            )
        ),
    )

    response_synthesizer = TreeSummarize(
        service_context=service_context,
        text_qa_template=DEFAULT_TEXT_QA_PROMPT,
        streaming=False,
    )

    response = response_synthesizer.get_response(
        query_str=args.query,
        text_chunks=[args.text_chunk]
    )
    print(response)
    return


if __name__ == '__main__':
    main()
