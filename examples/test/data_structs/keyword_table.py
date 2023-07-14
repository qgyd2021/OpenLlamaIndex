#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

from llama_index.data_structs.data_structs import KeywordTable
from llama_index.schema import Document


DEFAULT_TEXT = """
我们是谁，我们在做什么
牛信云是一家专业提供CPaaS、SaaS等多种接入方式的国际云通信公司。
自成立以来，专注于云通信领域的技术研发，陆续推出了国际短信、国际语音、全球号码、WhatsApp Business、Viber Business等CPaaS服务，以及NXLink、NXCallbot、NXCC等SaaS服务。
目前，牛信云中国区总部坐落在深圳，并在宁波、上海、成都、厦门、香港设立分公司；
国际总部位于新加坡。深度合作客户有阿里巴巴、腾讯、小米、欢聚时代等头部互联网企业，旨在为出海企业提供一站式通信解决方案，帮助更多企业走向世界。
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT, type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    index_struct = KeywordTable()

    document = Document(
        text=args.text,
        metadata={
            "source": "metadata_source",
        }
    )
    index_struct.add_node(
        keywords=["牛信"],
        node=document,
    )

    print(index_struct.keywords)
    print(index_struct.node_ids)
    return


if __name__ == '__main__':
    main()
