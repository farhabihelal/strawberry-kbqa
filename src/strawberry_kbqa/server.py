import json
import logging
import logging.config

from flask import Flask, request as Request

from qa import QAHandler


class QAService:

    def __init__(self, config: dict):
        self.configure(config)

        self.server = Flask(__name__)

        self.setup_routes()
        self.setup_qa_handler()

    def configure(self, config: dict):
        self.config = config

        self.port = config.get("port", 9880)

    def setup_qa_handler(self):
        self.qa_handler = QAHandler(self.config)

    def setup_routes(self):
        @self.server.route("/kb/qa", methods=["POST"])
        def answer() -> str:
            request = Request.get_json()
            logging.info(f"Received request: {request}")
            answer = self.qa_handler.answer(request["question"], request["context"])

            response = {
                "answer": answer,
            }
            return json.dumps(response)

    def run(self, port: int = None):
        port = port or self.port
        logging.info(f"Starting QA service on port {port}")
        self.server.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    config = {
        "port": 9880,
    }
    service = QAService(config)
    service.run()
