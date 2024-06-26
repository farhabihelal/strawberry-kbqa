import json
import logging
import os
import re
import sys
import threading
from typing import Any, Iterable, List

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import InMemoryStore, LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import Runnable
from timer import Timer, measure_time


class BasePipeline:
    def __init__(self, config: dict) -> None:

        self.configure(config)

        self.llm = self.create_llm()
        self.prompt = self.create_prompt()
        self.chain = self.create_chain()

        self.thread = None
        self.raw_result = None

        self.success = False

    def configure(self, config: dict) -> None:
        logging.debug(f"Configuring with: `{config}` ...")
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

    def process_response(self, response: Any) -> str:
        result = response if type(response) is str else response["answer"]
        return result

    @measure_time
    def _invoke(self, query: dict, chain: Runnable) -> None:
        response = chain.invoke(query)
        self.raw_result = self.process_response(response)
        self.success = not self.has_failed()

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
        return any(
            [
                x in self.raw_result
                for x in ["sorry", "does not", "not", "cannot", "unable"]
            ]
        )

    def join(self) -> None:
        self.thread.join()

    @property
    def result(self) -> str:
        return self.raw_result if self.success else ""


class ResponseValidationPipeline(BasePipeline):

    default_model_name = "llama2"
    default_prompt_template = """<s>[INST]Process the given sentence based on the steps in order:
    
    step: Remove any questions from the sentence.
    For example, The sentence: "I am Haru. How are you today?" should be converted to "I am Haru."
    
    step: Return the processed sentence.

    </s>[/INST]
    
    [INST]Sentence: {input}[/INST]
    """

    def __init__(self, config: dict = None) -> None:
        config = config or self.get_default_config()
        super().__init__(config)

    def configure(self, config: dict) -> None:
        if "model_name" not in config:
            config["model_name"] = Pipeline.default_model_name

        if "prompt_template" not in config:
            config["prompt_template"] = Pipeline.default_prompt_template

        return super().configure(config)

    @classmethod
    def get_default_config(self) -> dict:
        return {
            "model_name": ResponseValidationPipeline.default_model_name,
            "prompt_template": ResponseValidationPipeline.default_prompt_template,
        }

    # def create_prompt(self, prompt_template: str = None) -> SystemMessagePromptTemplate:
    #     prompt_template = prompt_template or self.prompt_template
    #     prompt = SystemMessagePromptTemplate.from_template(prompt_template)

    #     return prompt


class Pipeline(BasePipeline):

    default_model_name = "mistral"
    default_prompt_template = """Answer the following question based on general knowledge and common sense.
Follow these instructions:
- Use less than three sentences, preferably one.
- Be polite and friendly.
- The answer should be short, polite, and succinct.
- Never ask questions.
- Convert units if necessary. But only tell the final answer. No need to explain the steps.
- Assume units in SI unless otherwise specified.

Here are some examples of general knowledge questions and answer formats:
Q: What is the capital of France?
A: The capital of France is Paris.
Q: What is the population of Japan?
A: The population of Japan is 126.3 million.
Q: What is the currency of India?
A: The currency of India is the Indian rupee.
Q: What is the speed of light?
A: The speed of light is 299,792,458 meters per second.

Question: {input}"""

    def __init__(self, config: dict = None) -> None:
        config = config or self.get_default_config()
        super().__init__(config)

    def configure(self, config: dict) -> None:
        if "model_name" not in config:
            config["model_name"] = Pipeline.default_model_name

        if "prompt_template" not in config:
            config["prompt_template"] = Pipeline.default_prompt_template

        return super().configure(config)

    @classmethod
    def get_default_config(self) -> dict:
        return {
            "model_name": Pipeline.default_model_name,
            "prompt_template": Pipeline.default_prompt_template,
        }

    def run(self, query: dict, *args, **kwargs) -> None:
        return super().run(query)


class RAGPipeline(BasePipeline):
    default_embeddings = OllamaEmbeddings()
    default_model_name = "llama2"
    default_prompt_template = """You are Haru, a social robot. Answer the following question based only on the provided context.
Follow these instructions:
- Use less than three sentences, preferably one.
- Be polite and friendly.
- Never mention your context in the response.
- Never say the source.
- The answer should be short, polite, and succinct.
- Do not unnecessarily remind people that who you are in your answer.
- Never ask questions.
- Convert units if necessary. But only tell the final answer. No need to explain the steps.
- Assume units in SI unless otherwise specified.

Here are some examples of general knowledge questions and answer formats:
Q: What is the capital of France?
A: The capital of France is Paris.
Q: What is the population of Japan?
A: The population of Japan is 126.3 million.
Q: What is the currency of India?
A: The currency of India is the Indian rupee.
Q: What is the speed of light?
A: The speed of light is 299,792,458 meters per second.



<context>
{context}
</context>

Question: {input}"""

    def __init__(self, config: dict = None) -> None:
        config = config or self.get_default_config()
        super().__init__(config)

        self.current_embeddings = RAGPipeline.default_embeddings

        self.local_store = InMemoryStore()

        self.cached_embedder = self.create_cache(RAGPipeline.default_embeddings)

    def configure(self, config: dict) -> None:
        if "model_name" not in config:
            config["model_name"] = RAGPipeline.default_model_name

        if "prompt_template" not in config:
            config["prompt_template"] = RAGPipeline.default_prompt_template

        return super().configure(config)

    def create_documents(self, data: str) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter()
        documents = [Document(page_content=data)]
        splitted_documents = text_splitter.split_documents(documents)
        return splitted_documents or documents

    def create_chain(
        self, llm: Ollama = None, prompt: ChatPromptTemplate = None
    ) -> Runnable:
        llm = llm or self.llm
        prompt = prompt or self.prompt

        return create_stuff_documents_chain(llm, prompt)

    def create_cache(self, embeddings: Embeddings) -> CacheBackedEmbeddings:
        self.current_embeddings = embeddings
        return CacheBackedEmbeddings.from_bytes_store(
            embeddings, self.local_store, namespace=embeddings.model
        )

    def create_retrieval_chain(
        self, context: str, embeddings: Embeddings = None
    ) -> Runnable:
        embeddings = embeddings or self.current_embeddings

        if embeddings != self.current_embeddings:
            logging.info("Creating new cache ...")
            self.cached_embedder = self.create_cache(embeddings)

        documents = self.create_documents(context)
        db = FAISS.from_documents(documents, self.cached_embedder)

        return create_retrieval_chain(db.as_retriever(), self.chain)

    def run(self, query: dict, context: str) -> None:
        retrieval_chain = self.create_retrieval_chain(context)

        super().run(query, chain=retrieval_chain)

    @classmethod
    def get_default_config(self) -> dict:
        return {
            "model_name": RAGPipeline.default_model_name,
            "prompt_template": RAGPipeline.default_prompt_template,
        }


if __name__ == "__main__":
    # query = {
    #     "input": "how old is the universe and earth?",
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

    # answer = "The universe is 13.8 billion years old. The Earth is 4.5 billion years old. How old are you?"
    # query = {"input": answer}
    # rv_pipe = ResponseValidationPipeline()
    # rv_pipe.run(query)
    # rv_pipe.join()
    # print(rv_pipe.result)

    # pipe = Pipeline()

    # pipe.run(query, context={})
    # pipe.join()
    # print(pipe.result)

    pipe = RAGPipeline()

    context = """("person1", "hasName", "Timmy"),
("person1", "hasAge", "25"),
("person1", "hasPet", "cat"),
("person2", "hasName", "Jimmy"),
("person2", "hasAge", "35"),
("person2", "hasPet", "lion"),
("person2", "hasFriend", "person1"),
("global_context", "hasSubject", "Jimmy"),
"""

    context = context.replace("\n", "")

    query = {
        "input": "How old is he?",
    }
    pipe.run(query, "")
    pipe.join()
    print(pipe.result)

    query = {
        "input": "How old is Timmy?",
    }
    with Timer("RAGPipeline"):
        pipe.run(query, context)
        pipe.join()
    print(pipe.result)
