import requests

from rich.console import Console
from rich.prompt import Prompt

from client import QAClient


class CLIClient(QAClient):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

        self.console = Console()

    def display_message(self, sender: str, message: str) -> None:
        self.console.print(f"[{sender}] {message}\n")

    def run(self) -> None:
        self.console.print("Welcome to the CLI Chat App!", style="bold green")

        while True:
            message = Prompt.ask("Enter your message (type 'exit' to quit):")

            if message.lower() == "exit":
                self.console.print("Goodbye!")
                break

            response = self.send_request(message)
            self.display_message("Server", response)


if __name__ == "__main__":
    config = {
        "host": "localhost",
        "port": 9880,
        "context": {
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
        },
    }
    app = CLIClient(config)
    app.run()
