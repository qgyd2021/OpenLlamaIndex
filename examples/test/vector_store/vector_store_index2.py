#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.schema import TextNode

import project_settings as settings


def get_args():
    parser = argparse.ArgumentParser()
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
        embed_model=OpenAIEmbedding(api_key=args.openai_api_key),
    )

    index = VectorStoreIndex(
        nodes=[],
        service_context=service_context,
    )

    node1 = TextNode(text="<text_chunk>", id_="<node_id>")
    index.insert_nodes([node1])

    print(index.index_struct.nodes_dict)
    print(index.storage_context.docstore.docs)

    print(index.vector_store.is_embedding_query)
    return


if __name__ == '__main__':
    main()
