import json
import logging
import os
import sys
from typing import Any, Iterable, List

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


class QAHandler:

    def __init__(self, config: dict):
        self.configure(config)

        self.prompt_chain = None
        self.llm_chain = None
        self.document_chain = None

        self.prepare_chains()

    def configure(self, config: dict):
        self.config = config

        self.model_name = config.get("model", "mistral")

        self.character = config.get(
            "character",
            self.get_default_character(),
        )

        self.prompt_template = config.get(
            "prompt_template",
            self.get_default_prompt_template(),
        )

    def prepare_chains(self):
        self.prompt_chain = self.create_prompt_chain()
        self.llm_chain = self.create_llm_chain()
        self.document_chain = self.create_document_chain()

    def answer(self, question: str, context: str) -> str:
        retrieval_chain = self.create_retrieval_chain(context)
        response = retrieval_chain.invoke({"input": question})
        return response["answer"]

    def create_documents(self, data: str) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents([Document(page_content=data)])
        return documents

    def create_prompt_chain(self, prompt_template: str = None) -> ChatPromptTemplate:
        prompt_template = prompt_template or self.prompt_template
        return ChatPromptTemplate.from_template(prompt_template)

    def create_llm_chain(self, model_name: str = None) -> Any:
        model_name = model_name or self.model_name

        llm_chain = Ollama(model=model_name)
        return llm_chain

    def create_document_chain(self, llm_chain=None, prompt_chain=None) -> Runnable:
        llm_chain = llm_chain or self.llm_chain
        prompt_chain = prompt_chain or self.prompt_chain

        document_chain = create_stuff_documents_chain(llm_chain, prompt_chain)
        return document_chain

    def create_retrieval_chain(self, context: str, embeddings=None) -> Runnable:
        embeddings = embeddings or OllamaEmbeddings()
        documents = self.create_documents(context)
        vector = FAISS.from_documents(documents, embeddings)

        retriever = vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, self.document_chain)

        return retrieval_chain

    @classmethod
    def get_default_prompt_template(self) -> str:
        return """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}"""


if __name__ == "__main__":

    def process_context(context: list) -> str:
        return json.dumps(context)

    logging.basicConfig(level=logging.INFO)

    prompt_template = """Answer the following question based only on the provided context:
<context>
{context}
</context>

Question: {input}"""

    config = {
        "model": "mistral",
        "prompt_template": prompt_template,
    }
    handler = QAHandler(config)

    context_data = [
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
    ]

    context = process_context(context_data)

    while True:
        question = input("Enter a question: ")
        response = handler.answer(question, context)
        print(response)
