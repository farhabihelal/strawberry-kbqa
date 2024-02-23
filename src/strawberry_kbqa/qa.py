import json
import logging
import os
import sys
from typing import Any, Iterable, List

import pip

from pipeline import Pipeline, RAGPipeline


class QAHandler:

    def __init__(self, config: dict = None) -> None:
        self.configure(config)

        self.pipelines = []

        self.setup_pipelines()

    def configure(self, config: dict):
        self.config = config

    def setup_pipelines(self):
        self.pipelines.append(RAGPipeline(dict(self.config)))

        self.pipelines.append(Pipeline(dict(self.config)))

    def answer(self, question: str, triple_data: dict) -> str:
        context = self.convert_triple2context(triple_data["triples"])
        context = self.process_context(context)
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

        return response

    @staticmethod
    def convert_triple2context(triples: list) -> list:
        context = [list(x.values()) for x in triples]
        return context

    @staticmethod
    def process_context(context: list) -> str:
        processed_context = []

        unique_persons = set([x[0] for x in context if "person" in x[0]])
        person_names = {
            person: [x[2] for x in context if x[0] == person and "hasName" in x[1]][0]
            for person in unique_persons
        }

        for triple in context:
            subject, predicate, object = triple

            if any(x in predicate for x in ["hasName", "rdf:type"]):
                continue

            if subject in unique_persons:
                subject = person_names[subject]
            if object in unique_persons:
                object = person_names[object]

            processed_context.append((subject, predicate, object))

        processed_context = "\n".join(
            [f'"{x[0]}","{x[1]}","{x[2]}"' for x in processed_context]
        )

        return processed_context

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

    @classmethod
    def get_default_prompt_template(self) -> str:
        return """Answer the following question based only on the provided context:
<context>
{context}
</context>

Question: {input}"""


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
