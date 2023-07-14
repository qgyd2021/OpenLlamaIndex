#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os

from llama_index.storage.storage_context import StorageContext
from llama_index.storage.docstore.types import BaseDocumentStore
from llama_index.storage.index_store.types import BaseIndexStore
from llama_index.vector_stores.types import VectorStore
from llama_index.graph_stores.types import GraphStore

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--persist_dir",
        default=(project_path / "cache/persist_dir").as_posix(),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    storage_context: StorageContext = StorageContext.from_defaults()
    print(storage_context)

    doc_store: BaseDocumentStore = storage_context.docstore
    index_store: BaseIndexStore = storage_context.index_store
    vector_store: VectorStore = storage_context.vector_store
    graph_store: GraphStore = storage_context.graph_store
    print(doc_store)
    print(index_store)
    print(vector_store)
    print(graph_store)

    os.makedirs(args.persist_dir, exist_ok=True)
    storage_context.persist(persist_dir=args.persist_dir)

    return


if __name__ == '__main__':
    main()
