#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from typing import List

from llama_index.schema import Document
from llama_index import SimpleDirectoryReader

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default=(project_path / "cache/arxiv_pdf").as_posix(),
        type=str
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    directory_reader = SimpleDirectoryReader(input_dir=args.input_dir)
    documents: List[Document] = directory_reader.load_data()

    for document in documents:
        print(document)
    return


if __name__ == '__main__':
    main()
