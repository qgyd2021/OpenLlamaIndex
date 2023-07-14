#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

from llama_index.indices.query.schema import QueryBundle

import project_settings as settings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default="what is instruction GPT ?",
        type=str
    )
    parser.add_argument(
        "--openai_api_key",
        default=settings.environment.get("openai_api_key", default=None, dtype=str),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    return


if __name__ == '__main__':
    main()
