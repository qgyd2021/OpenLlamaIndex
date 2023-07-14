#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from typing import List

from llama_index.schema import Document
from llama_index.readers.file.docs_reader import PDFReader

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        default=(project_path / "cache/arxiv_pdf/2203.02155.pdf").as_posix(),
        type=str
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    reader = PDFReader()

    file = args.filename

    documents: List[Document] = reader.load_data(file=Path(file))
    print(documents)
    return


if __name__ == '__main__':
    main()
