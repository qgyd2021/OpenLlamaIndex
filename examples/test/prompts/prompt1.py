#!/usr/bin/python3
# -*- coding: utf-8 -*-
from llama_index.prompts.base import Prompt
from llama_index.prompts.prompts import QuestionAnswerPrompt, SummaryPrompt
from llama_index.prompts.prompt_type import PromptType


def main():

    prompt_template = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the question: {query_str}\n"
    )

    prompt = Prompt(
        prompt_template, prompt_type=PromptType.QUESTION_ANSWER
    )
    print(prompt.prompt)
    print(prompt.partial_dict)

    prompt = prompt.partial_format(query_str="hello")
    print(prompt.prompt)
    print(prompt.partial_dict)

    prompt = SummaryPrompt.from_prompt(
        prompt, prompt_type=PromptType.SUMMARY
    )
    print(prompt.prompt)
    print(prompt.partial_dict)

    return


if __name__ == '__main__':
    main()
