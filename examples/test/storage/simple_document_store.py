#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json

from llama_index.schema import Document, MetadataMode
from llama_index.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.storage.docstore.simple_docstore import SimpleDocumentStore, SimpleKVStore

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
    parser.add_argument("--text", default=DEFAULT_TEXT, type=str)
    parser.add_argument(
        "--persist_path",
        default=(project_path / "cache/persist_dir/simple_kv_store.json").as_posix(),
        type=str
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    simple_kv_store = SimpleKVStore()

    kv_document_store = KVDocumentStore(
        kvstore=simple_kv_store,
    )

    document = Document(
        text=args.text,
        metadata={
            "source": "metadata_source",
        }
    )

    kv_document_store.add_documents(
        nodes=[document]
    )

    document_store = SimpleDocumentStore(
        simple_kvstore=kv_document_store,
        namespace="namespace"
    )

    # only support BaseInMemoryKVStore
    document_store.persist(persist_path=args.persist_path)
    # persist
    simple_kv_store.persist(persist_path=args.persist_path)

    # from_persist_path
    document_store = SimpleDocumentStore.from_persist_path(persist_path=args.persist_path)
    document_store_dict = document_store.to_dict()
    document_store_dict_str = json.dumps(document_store_dict, ensure_ascii=False, indent=4)
    print(document_store_dict_str)
    return


if __name__ == '__main__':
    main()
