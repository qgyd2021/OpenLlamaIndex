#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

from llama_index.schema import IndexNode


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_id", default=0, type=int)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    node = IndexNode(index_id=args.index_id)
    print(node)
    return


if __name__ == '__main__':
    main()
