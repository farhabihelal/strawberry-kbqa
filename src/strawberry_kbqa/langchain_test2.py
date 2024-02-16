import os
import sys

import threading

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader, WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.documents import Document

embeddings = OllamaEmbeddings()


data = """
[
("person1", "hasName", "Timmy"),
("person1", "hasAge", "25"),
("person1", "hasPet", "cat"),
("person2", "hasName", "Jimmy"),
("person2", "hasAge", "35"),
("person2", "hasPet", "lion"),
("person2", "hasFriend", "person1"),
]
"""
docs = [Document(page_content=data)]

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)


llm = Ollama(model="mistral")

prompt_plain = ChatPromptTemplate.from_template(
    """Answer the following question based on general knowledge and common sense. Use less than 2 sentences. Be polite and friendly. Say "sorry" if you don't know the answer.
Question: {input}"""
)

prompt_doc = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context. Use less than 3 sentences. Be polite and friendly. Never mention context in response.

<context>
{context}
</context>

Question: {input}"""
)
document_chain = create_stuff_documents_chain(llm, prompt_doc)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


output_parser = StrOutputParser()

plain_chain = prompt_plain | llm | output_parser


def get_response(chain):
    response = chain.invoke({"input": question})
    response = response if type(response) == str else response["answer"]
    print(f"\nresponse: {response}\n")


while True:
    question = input("Enter a question: ")

    th_plain = threading.Thread(target=get_response, args=[plain_chain])
    th_doc = threading.Thread(target=get_response, args=[retrieval_chain])

    th_plain.start()
    th_doc.start()

    th_plain.join()
    th_doc.join()
