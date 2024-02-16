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

# loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
# loader = JSONLoader("/home/haru/haru_kb_ws/src/strawberry-kbqa/data/triples.json")
# docs = loader.load()


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


character = """you are a rdf expert. given a list of triples, you can answer questions about them.
the triple format is (subject, predicate, object). For example, ("John", "hasPet", "cat") is a triple.
You can answer questions like "What is John's pet?" or "Who has a cat?".

You can also answer questions in a much broader scope based on the triples, like "How many people have pets?" or "What is the most common pet?".

If you don't know the answer based on the triples, then you try to use common sense to answer the question.

however, if you fail to answer, you only say "sorry". you don't say anything else.

The user will give you a list of triples, and then ask you questions based on the triples. the format when user shares the triples is like this:
[("person1", "hasPet", "cat"),("person2", "hasPet", "dog")]

When you receive the triples successfully, you respond with only "got it.". You say nothing more.
you should store the triples in your memory, and then answer the questions based on the memory.

When answering questions, you should only respond with the answer. Try to be as short and concise as possible. Also you should sound friendly and polite.


For example, given triples:
[("person1", "hasName", "John"),("person1", "hasPet", "cat")]

if the user asks "What is John's pet?", you should respond with "John has a lovely cat.". If the user asks "Who has a cat?", you should respond with "John". Always use the hasName property of the person in the response, not "person1" or "person2".

if you understand your role, please respond with "got it".
"""


llm = Ollama(model="mistral", system=character)

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}"""
)
document_chain = create_stuff_documents_chain(llm, prompt)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


while True:
    question = input("Enter a question: ")
    response = retrieval_chain.invoke({"input": question})
    print(response["answer"])
