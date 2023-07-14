#!/usr/bin/python3
# -*- coding: utf-8 -*-
from llama_index.prompts.base import Prompt
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.prompts.prompts import QuestionAnswerPrompt, SummaryPrompt
from llama_index.prompts.prompt_type import PromptType
from llama_index.indices.prompt_helper import PromptHelper


def main():

    prompt_helper = PromptHelper()

    prompt = DEFAULT_TEXT_QA_PROMPT.partial_format(query_str="hello")
    print(prompt.prompt)
    print(prompt.partial_dict)

    prompt = SummaryPrompt.from_prompt(
        prompt, prompt_type=PromptType.SUMMARY
    )

    text_splitter = prompt_helper.get_text_splitter_given_prompt(prompt, padding=5)
    print(text_splitter)
    return


if __name__ == '__main__':
    main()
