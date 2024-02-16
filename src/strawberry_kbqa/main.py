import sys
import os

from time import sleep

from ollama import Client

client = Client(host="http://localhost:11434")

model = "mistral"
messages = []

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

character = character.replace('"', '\\"').replace("\n", "\\n")
print(character)

messages.append({"role": "user", "content": character})
response = client.chat(
    model=model,
    messages=messages,
    stream=False,
)
print(response["message"])
# messages.append(response["message"])

triples = [
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

triple_str = "here are the triples:"
triple_str += str(triples).strip().replace(", ", ",").replace("\n", "")

messages.append({"role": "user", "content": triple_str})
response = client.chat(
    model=model,
    messages=messages,
    stream=False,
)

print(response["message"])
# messages.append(response["message"])

question = "What is Timmy's pet?"

messages.append({"role": "user", "content": question})
response = client.chat(
    model=model,
    messages=messages,
    stream=False,
)
print(response["message"])
# messages.append(response["message"])

print(messages)
