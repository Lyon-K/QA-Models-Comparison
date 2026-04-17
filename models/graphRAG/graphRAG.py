import os
from typing import Literal, Optional
from neo4j import GraphDatabase, Driver, exceptions
from langchain_huggingface import HuggingFaceEmbeddings
from ollama import Client
import logging

logger = logging.getLogger(__name__)


class GraphRAG:
    driver: Optional[Driver]
    embedding_model: HuggingFaceEmbeddings
    llm_model: Client

    def __init__(
        self,
        embedding_model: Optional[HuggingFaceEmbeddings] = None,
        llm_model: Optional[Client] = None,
    ):
        logger.info("Starting graphRAG...")
        self._URI = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        if not all([self._URI, user, password]):
            raise ValueError("Neo4j credentials missing")
        self._AUTH = (user, password)

        try:
            driver = GraphDatabase.driver(self._URI, auth=self._AUTH)
            driver.verify_connectivity()
            self.driver = driver

            self.embedding_model = embedding_model or self._default_embedding()
            self.llm_model = llm_model or self._default_llm()

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            logger.info("Closing graphRAG...")
            self.close()
            raise

    def _default_embedding(self):
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            query_encode_kwargs={
                "prompt": "Represent this sentence for searching relevant passages: "
            },
        )

    def _default_llm(self):
        api_key = os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            raise ValueError("OLLAMA_API_KEY is not set")
        return Client(
            host="https://ollama.com", headers={"Authorization": "Bearer " + api_key}
        )

    def predict(self, query, n_hop: Literal[0, 1] = 0):
        context = self._rag_retrieval(query)
        logger.debug(f"Query: {query}")
        logger.debug(f"n-hops: {n_hop}")
        logger.debug(
            "\n".join(
                [
                    f'score: {record["score"]:.3f} | {record["text"]}'
                    for record in context
                ]
            )
        )

        prompt = (
            f"**context(only use when it is relevant to the prompt given)**:\n{context}\n\n**prompt**:\n{query}"
            if context
            else query
        )
        response = self.llm_model.chat(
            # model="ministral-3:3b-cloud",
            model="ministral-3:8b-cloud",
            # model="ministral-3:14b-cloud",
            # model="gpt-oss:20b-cloud",
            # model="gpt-oss:120b-cloud",
            # model="mistral-large-3:675b-cloud",
            messages=[{"role": "user", "content": prompt}],
        )
        return context, response.message.content

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None

    def _rag_retrieval(self, query, n_hop=0, top_k=3):
        self.check_db()
        graph_query = [
            """MATCH ()-[r]->()
WHERE r.embedding IS NOT NULL
WITH r, gds.similarity.cosine(r.embedding, $query_embedding) AS score
RETURN r.embedding_text AS text, score
ORDER BY score DESC
LIMIT $top_k
""",
            """MATCH (a)-[r]->(b)
WHERE r.embedding IS NOT NULL
WITH a, r, b, gds.similarity.cosine(r.embedding, $query_embedding) AS score
ORDER BY score DESC
LIMIT $top_k

OPTIONAL MATCH (b)-[r2]->(c)
RETURN r.embedding_text AS text, score""",
        ][n_hop]
        query_embedding = self.embedding_model.embed_query(query)
        with self.driver.session() as session:
            result = session.run(
                graph_query,
                {"query_embedding": query_embedding, "top_k": top_k},
            )
            context = result.data()
        return context

    def check_db(self):
        if not self.driver:
            raise exceptions.ServiceUnavailable(
                "GraphRAG cannot be used without a valid db connection"
            )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.DEBUG)

    graph_rag = GraphRAG()
    query = "What is the recommended daily sugar intake?"
    for n_hop in range(2):
        context, message = graph_rag.predict(query=query, n_hop=n_hop)
        print("Message:", message)
