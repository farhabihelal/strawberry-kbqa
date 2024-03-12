import json
import logging
import os
import sys
from typing import Any, Iterable, List

from context import Context, ContextData
from nlp import NLP
from pipeline import Pipeline, RAGPipeline


class QAHandler:

    def __init__(self, config: dict = None) -> None:
        self.configure(config)

        self.pipelines = []

        self.response_history = []

        self.setup_pipelines()

    def configure(self, config: dict):
        self.config = config

    def setup_pipelines(self):
        self.pipelines.append(RAGPipeline(dict(self.config)))

        self.pipelines.append(Pipeline(dict(self.config)))

    def answer(self, question: str, triple_data: dict) -> str:
        context_data = ContextData.from_triples(triple_data)
        context = str(context_data.to_compact_form())

        query = {
            "input": question,
        }

        logging.info(f"Answering question: {question}")
        for pipeline in self.pipelines:
            pipeline.run(query, context=context)

        logging.info("Waiting for pipelines to finish...")
        for pipeline in self.pipelines:
            pipeline.join()

        raw_response = ""
        for pipeline in self.pipelines:
            if pipeline.success:
                raw_response = pipeline.result
                break

        response = self.filter_answer(raw_response)

        self.response_history.append(response)

        return response

    @staticmethod
    def filter_answer(raw_answer: str) -> str:

        if not "context" in raw_answer:
            return raw_answer

        filtered_answer = raw_answer

        unwanted_texts = [
            "According to the context provided, ",
            "Based on the context provided, ",
            "Based on the provided context, ",
        ]

        for unwanted_text in unwanted_texts:
            filtered_answer = filtered_answer.replace(unwanted_text, "").strip()

        return filtered_answer


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    config = {}
    handler = QAHandler(config)

    context = [
        # peron1
        ("person1", "hasName", "Timmy"),
        ("person1", "hasAge", "25"),
        ("person1", "hasProfession", "Mechanic"),
        ("person1", "hasHometown", "California"),
        ("person1", "hasFriend", "person2"),
        ("person1", "hasPet", "cat"),
        # person2
        ("person2", "hasName", "Tommy"),
        ("person2", "hasAge", "26"),
        ("person2", "hasProfession", "Mechanic"),
        ("person2", "hasHometown", "New York"),
        ("person2", "hasFriend", "person1"),
        ("person2", "hasPet", "cat"),
        # haru
        ("haru", "hasName", "Haru"),
        ("haru", "hasAge", "7"),
        ("haru", "hasProfession", "Social Mediator"),
        ("haru", "hasHomeCountry", "Japan"),
        ("haru", "hasFavorite", "Baseball"),
        ("haru", "hasFavorite", "Fall"),
        ("haru", "hasFavorite", "Microchip"),
    ]

    print(json.dumps(QAHandler.process_context(context)))

    while True:
        question = input("Enter a question: ")
        response = handler.answer(question, context)
        print(response)
