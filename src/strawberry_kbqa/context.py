import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class Context:
    triple: List[str]

    def __post_init__(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.triple)

    def __getitem__(self, idx) -> str:
        return self.triple[idx]

    def __str__(self) -> str:
        return str(self.triple)

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def from_triple(triple: dict):
        return Context([x for x in triple.values()])

    def is_type(self) -> bool:
        return self.triple[1].lower() == "rdf:type"


@dataclass
class ContextData:
    contexts: List[Context]

    def __post_init__(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx) -> Context:
        return self.contexts[idx]

    def __repr__(self) -> str:
        output = f"ContextData with {len(self)} contexts.\n"
        for idx, context in enumerate(self.contexts):
            output += f"{idx:5}: {context.triple}\n"
        return output

    def __str__(self) -> str:
        return repr(self)

    @staticmethod
    def from_triples(triples: dict):
        triples = triples["triples"]

        contexts = []
        for triple in triples:
            context = Context.from_triple(triple)
            contexts.append(context)

        return ContextData(contexts)

    def get_unique_subjects(self) -> List[str]:
        subjects = []
        for context in self.contexts:
            subjects.append(context.triple[0])
        return list(set(subjects))

    def get_unique_predicates(self) -> List[str]:
        predicates = []
        for context in self.contexts:
            predicates.append(context.triple[1])
        return list(set(predicates))

    def get_all_favorites(self) -> List[Context]:
        return self.get_all_with_predicate("hasFavorite")

    def get_all_types(self) -> List[Context]:
        return [context for context in self.contexts if context.is_type()]

    def get_all_with_subject(self, subject: str) -> List[Context]:
        return [context for context in self.contexts if context.triple[0] == subject]

    def get_all_with_predicate(self, predicate: str) -> List[Context]:
        return [context for context in self.contexts if context.triple[1] == predicate]

    def get_all_with_object(self, object: str) -> List[Context]:
        return [context for context in self.contexts if context.triple[2] == object]

    def to_list_simple(self) -> List[List[str]]:
        return [x.triple for x in self.contexts]

    def to_list_compact(self) -> List[List[str]]:
        context_data = self
        context_data = ContextData.merge_types(context_data)
        context_data = ContextData.replace_subjects(context_data)

        return context_data.to_list_simple()

    @staticmethod
    def merge_types(context_data):
        all_types_map = {x[0]: x[2] for x in context_data.get_all_types()}

        return ContextData(
            [
                Context(
                    [
                        x[0],
                        f"{x[1]}{all_types_map.get(x[2], '')}",
                        x[2],
                    ]
                )
                for x in context_data
                if not x.is_type()
            ]
        )

    @staticmethod
    def replace_subjects(context_data):
        names_map = {x[0]: x[2] for x in context_data.get_all_with_predicate("hasName")}

        return ContextData(
            [
                Context(
                    [
                        f"{names_map.get(x[0], x[0])}",
                        x[1],
                        f"{names_map.get(x[2], x[2])}",
                    ]
                )
                for x in context_data
                if not x.is_type()
            ]
        )


if __name__ == "__main__":

    raw_context_data = {
        "triples": [
            # peron1
            {"subject": "person1", "predicate": "hasName", "object": "Timmy"},
            {"subject": "person1", "predicate": "hasAge", "object": "25"},
            {
                "subject": "person1",
                "predicate": "hasProfession",
                "object": "Mechanic",
            },
            {
                "subject": "person1",
                "predicate": "hasHometown",
                "object": "California",
            },
            {"subject": "person1", "predicate": "hasFriend", "object": "person2"},
            {"subject": "person1", "predicate": "hasPet", "object": "cat"},
            # person2
            {"subject": "person2", "predicate": "hasName", "object": "Tommy"},
            {"subject": "person2", "predicate": "hasAge", "object": "26"},
            {
                "subject": "person2",
                "predicate": "hasProfession",
                "object": "Mechanic",
            },
            {
                "subject": "person2",
                "predicate": "hasHometown",
                "object": "New York",
            },
            {"subject": "person2", "predicate": "hasFriend", "object": "person1"},
            {"subject": "person2", "predicate": "hasPet", "object": "bird"},
            # haru
            {"subject": "haru", "predicate": "hasName", "object": "Haru"},
            {"subject": "haru", "predicate": "hasAge", "object": "7"},
            {
                "subject": "haru",
                "predicate": "hasProfession",
                "object": "Social Mediator",
            },
            {"subject": "haru", "predicate": "hasHomeCountry", "object": "Japan"},
            {"subject": "Japan", "predicate": "rdf:type", "object": "Country"},
            {"subject": "haru", "predicate": "hasFavorite", "object": "Baseball"},
            {"subject": "Baseball", "predicate": "rdf:type", "object": "Sports"},
            {"subject": "haru", "predicate": "hasFavorite", "object": "Fall"},
            {"subject": "Fall", "predicate": "rdf:type", "object": "Season"},
            {"subject": "haru", "predicate": "hasFavorite", "object": "Microchips"},
            {"subject": "Microchips", "predicate": "rdf:type", "object": "Food"},
            {"subject": "haru", "predicate": "hasFavorite", "object": "Godzilla"},
            {"subject": "Godzilla", "predicate": "rdf:type", "object": "Movie"},
            {
                "subject": "haru",
                "predicate": "hasFavorite",
                "object": "Watching_TV",
            },
            {"subject": "Watching_TV", "predicate": "rdf:type", "object": "Hobby"},
            {"subject": "haru", "predicate": "hasFavorite", "object": "Asia"},
            {"subject": "Asia", "predicate": "rdf:type", "object": "Continent"},
            {"subject": "haru", "predicate": "hasFavorite", "object": "North"},
            {"subject": "North", "predicate": "rdf:type", "object": "Hemisphere"},
        ]
    }

    context_data = ContextData.from_triples(raw_context_data)
    # print(context_data)
    # print(context_data.get_all_unique_subjects())
    # print(context_data.get_all_unique_predicates())
    # print(context_data.get_all_favorites())
    # print(context_data.get_all_with_subject("haru"))
    # print(context_data.to_list_simple())
    print(context_data.to_list_compact())
    # print(context_data.get_type_appended_favorites())
    # print(context_data.merge_types())
