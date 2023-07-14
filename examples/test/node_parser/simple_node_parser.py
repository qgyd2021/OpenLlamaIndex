#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from typing import List

from llama_index.schema import Document, TextNode
from llama_index.node_parser.simple import SimpleNodeParser


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default="document one. test the simple node parser. ",
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    documents = [
        Document(text=args.text)
    ]

    parser = SimpleNodeParser()

    nodes: List[TextNode] = parser.get_nodes_from_documents(documents)
    print(nodes)
    return


if __name__ == '__main__':
    main()
