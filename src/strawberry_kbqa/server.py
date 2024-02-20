import json
import logging
import logging.config

from flask import Flask

from qa import QAHandler


class QAService:

    def __init__(self, config: dict):
        self.configure(config)

        self.app = Flask(__name__)

        self.setup_routes()
        self.setup_qa_handler()

    def configure(self, config: dict):
        self.config = config

        self.port = config.get("port", 9880)
        self.model = config.get("model", "mistral")

    def setup_qa_handler(self):
        self.qa_handler = QAHandler(self.config)

    def setup_routes(self):
        @self.app.route("/strawberry_kb/qa", methods=["POST"])
        def answer(request: str) -> str:
            logging.info(f"Received request: {request}")
            request = json.loads(request)
            answer = self.qa_handler.answer(request["question"], request["context"])

            response = {
                "answer": answer,
            }
            return json.dumps(response)

    def run(self, port: int = None):
        port = port or self.port
        logging.info(f"Starting QA service on port {port}")
        self.app.run(port=port)


if __name__ == "__main__":
    config = {
        "port": 9880,
        "model": "mistral",
    }
    service = QAService(config)
    service.run()
