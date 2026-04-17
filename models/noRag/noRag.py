from ollama import Client


class NoRAG:
    llm_model: Client

    def __init__(self, llm_model=None):
        self.llm_model = llm_model

    def predict(self, query):
        # Implement Prediction
        response = self.llm_model.chat(
            # model="ministral-3:3b-cloud",
            model="ministral-3:8b-cloud",
            # model="ministral-3:14b-cloud",
            # model="gpt-oss:20b-cloud",
            # model="gpt-oss:120b-cloud",
            # model="mistral-large-3:675b-cloud",
            messages=[{"role": "user", "content": query}],
        )
        return None, response.message.content
