import json
import logging
import os
import sys
import threading
from typing import Any, Iterable, List

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


class BasePipeline:
    def __init__(self, config: dict) -> None:

        self.configure(config)

        self.llm = self.create_llm()
        self.prompt = self.create_prompt()
        self.chain = self.create_chain()

        self.thread = None
        self.result = None

    def configure(self, config: dict):
        self.config = config

        self.model_name = config["model_name"]
        self.prompt_template = config["prompt_template"]

    def create_chain(
        self, llm: Ollama = None, prompt: ChatPromptTemplate = None
    ) -> Runnable:
        llm = llm or self.llm
        prompt = prompt or self.prompt

        return prompt | llm

    def create_llm(self, model_name: str = None) -> Ollama:
        model_name = model_name or self.model_name
        llm = Ollama(model=model_name)

        return llm

    def create_prompt(self, prompt_template: str = None) -> ChatPromptTemplate:
        prompt_template = prompt_template or self.prompt_template
        prompt = ChatPromptTemplate.from_template(prompt_template)

        return prompt

    def process_response(self, response: Any) -> Any:
        self.result = response if type(response) is str else response["answer"]

    def _invoke(self, query: dict, chain: Runnable) -> None:
        response = chain.invoke(query)
        self.process_response(response)

    def _run(self, *args, **kwargs) -> None:
        th = threading.Thread(target=self._invoke, args=[*args], kwargs={**kwargs})
        th.start()

        self.thread = th

    def run(self, query: dict, chain: Runnable = None) -> None:
        chain = chain or self.chain
        self._run(query=query, chain=chain)

    def is_running(self) -> bool:
        return self.thread.is_alive()

    def has_failed(self) -> bool:
        return True if "sorry" in self.result else False

    def join(self) -> None:
        self.thread.join()


class Pipeline(BasePipeline):

    default_model_name = "mistral"
    default_prompt_template = """Answer the following question based on general knowledge and common sense. Use less than 2 sentences. Be polite and friendly. Say "sorry" if you don't know the answer.
    Question: {question}"""

    def __init__(self, config: dict = None) -> None:
        config = config or self.get_default_config()
        super().__init__(config)

    @classmethod
    def get_default_config(self) -> dict:
        return {
            "model_name": Pipeline.default_model_name,
            "prompt_template": Pipeline.default_prompt_template,
        }


class RAGPipeline(BasePipeline):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

        self.default_embeddings = OllamaEmbeddings()
        self.current_embeddings = self.default_embeddings

        self.local_file_store_path = os.path.join("/", "tmp", "strawberry_qa", "cache")
        self.local_store = LocalFileStore(self.local_file_store_path)

        self.cached_embedder = self.create_cache(self.default_embeddings)

    def create_documents(self, data: str) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents([Document(page_content=data)])
        return documents

    def create_chain(
        self, llm: Ollama = None, prompt: ChatPromptTemplate = None
    ) -> Runnable:
        llm = llm or self.llm
        prompt = prompt or self.prompt

        return create_stuff_documents_chain(llm, prompt)

    def create_cache(self, embeddings: Embeddings) -> CacheBackedEmbeddings:
        return CacheBackedEmbeddings.from_bytes_store(
            embeddings, self.local_store, namespace=embeddings.model
        )

    def create_retrieval_chain(
        self, context: str, embeddings: Embeddings = None
    ) -> Runnable:
        embeddings = embeddings or self.default_embeddings

        if embeddings != self.current_embeddings:
            logging.info("Creating new cache ...")
            self.cached_embedder = self.create_cache(embeddings)

        documents = self.create_documents(context)
        vector = FAISS.from_documents(documents, self.cached_embedder)

        return create_retrieval_chain(vector.as_retriever(), self.chain)

    def run(self, query: dict, context: str) -> None:
        retrieval_chain = self.create_retrieval_chain(context)

        super().run(query, chain=retrieval_chain)


if __name__ == "__main__":
    # query = {
    #     "question": "how old is the universe and earth?",
    # }

    # prompt_template = """Answer the following question based on general knowledge and common sense. Use less than 2 sentences. Be polite and friendly. Say "sorry" if you don't know the answer.
    # Question: {question}"""

    # config = {
    #     "model_name": "mistral",
    #     "prompt_template": prompt_template,
    # }

    # base_pipe = BasePipeline(config)
    # base_pipe.run(query)
    # print(base_pipe.is_running())
    # base_pipe.join()
    # print(base_pipe.result)
    # print(base_pipe.has_failed())
    # print(base_pipe.is_running())

    # pipe = Pipeline()

    # pipe.run(query)
    # print(pipe.is_running())
    # pipe.join()
    # print(pipe.result)
    # print(pipe.has_failed())
    # print(pipe.is_running())

    prompt_template = """Answer the following question based only on the provided context. Use less than 3 sentences. Be polite and friendly. Never mention context in the response.
<context>
{context}
</context>

Question: {input}"""

    config = {
        "model_name": "mistral",
        "prompt_template": prompt_template,
    }
    pipe = RAGPipeline(config)

    context = """[
("person1", "hasName", "Timmy"),
("person1", "hasAge", "25"),
("person1", "hasPet", "cat"),
("person2", "hasName", "Jimmy"),
("person2", "hasAge", "35"),
("person2", "hasPet", "lion"),
("person2", "hasFriend", "person1"),
]"""

    context = context.replace("\n", "")

    query = {
        "input": "What is Timmy's pet?",
    }
    pipe.run(query, context)
    pipe.join()
    print(pipe.result)

    query = {
        "input": "How old is Timmy?",
    }
    pipe.run(query, context)
    pipe.join()
    print(pipe.result)
