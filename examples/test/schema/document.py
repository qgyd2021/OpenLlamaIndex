#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

from llama_index.schema import Document, MetadataMode
from langchain.schema.document import Document as LCDocument


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

    lc_document = LCDocument(
        page_content=args.text,
        metadata={
            "source": "metadata_source",
        }
    )

    document = Document.from_langchain_format(lc_document)
    print(document)

    content_all = document.get_content(metadata_mode=MetadataMode.ALL)
    print(content_all)

    metadata_str = document.get_metadata_str(mode=MetadataMode.ALL)
    print(metadata_str)

    print(document.get_doc_id())
    print(document.hash)
    return


if __name__ == '__main__':
    main()
