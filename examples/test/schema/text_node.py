#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

from llama_index.schema import MetadataMode, TextNode


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="text", type=str)
    parser.add_argument("--start_char_idx", default=0, type=int)
    parser.add_argument("--metadata_seperator", default="\n", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    text_node = TextNode(
        text=args.text,
        start_char_idx=args.start_char_idx,
        end_char_idx=args.start_char_idx + len(args.text),
        metadata_seperator=args.metadata_seperator,
        metadata={
            "source": "metadata source",
            "category": "water"
        }
    )
    print(text_node)

    content = text_node.get_content(metadata_mode=MetadataMode.ALL)

    print(content)
    print(type(content))
    print(len(content))

    return


if __name__ == '__main__':
    main()
