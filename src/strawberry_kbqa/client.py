import requests

from rich.console import Console
from rich.prompt import Prompt


class QAClient:
    def __init__(self, config: dict) -> None:
        self.configure(config)

        self.console = Console()

    def configure(self, config: dict) -> None:
        self.config = config

        self.host = config.get("host", "localhost")
        self.port = config.get("port", 9880)
        self.context = config.get("context", [])

    def display_message(self, sender: str, message: str) -> None:
        self.console.print(f"[{sender}] {message}\n")

    def send_request(self, message: str, context: list = None) -> str:
        context = context or self.context

        url = f"http://{self.host}:{self.port}/kb/qa"
        payload = {
            "question": message,
            "context": context,
        }

        response = requests.post(url, json=payload)
        return response.json()["answer"] if response.ok else "Error"

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
        "context": [
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
        ],
    }
    app = QAClient(config)
    app.run()
