#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from llama_index.indices.empty.base import EmptyIndex
from llama_index.indices.empty.retrievers import EmptyIndexRetriever


def get_args():
    return


def main():

    retriever = EmptyIndexRetriever(
        index=EmptyIndex()
    )

    result = retriever.retrieve("")
    print(result)
    return


if __name__ == '__main__':
    main()
