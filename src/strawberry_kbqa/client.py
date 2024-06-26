import requests


class QAClient:
    def __init__(self, config: dict) -> None:
        self.configure(config)

    def configure(self, config: dict) -> None:
        self.config = config

        self.host = config.get("host", "localhost")
        self.port = config.get("port", 9880)
        self.context = config.get("context", [])

    def send_request(self, message: str, context: list = None) -> str:
        context = context or self.context

        url = f"http://{self.host}:{self.port}/kb/qa"
        payload = {
            "question": message,
            "context": context,
        }

        response = requests.post(url, json=payload)
        return response.json()["answer"] if response.ok else "Server Error"


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
    client = QAClient(config)
    answer = client.send_request("What is your name?")
    print(answer)
