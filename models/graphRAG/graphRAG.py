import os
from typing import Optional
from neo4j import GraphDatabase, Driver, exceptions
from langchain_huggingface import HuggingFaceEmbeddings
from ollama import Client


class GraphRAG:
    driver: Driver
    driver_is_connected = False
    embedding_model: HuggingFaceEmbeddings
    model: Optional[Client] = None

    def __init__(
        self,
        embedding_model: Optional[HuggingFaceEmbeddings] = None,
        llm_model: Optional[Client] = None,
    ):
        print("Starting graphRAG...")
        try:
            self.driver = GraphDatabase.driver(
                uri=os.getenv("NEO4J_URI"),
                auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
            )
            self.driver.verify_connectivity()
            self.driver_is_connected = True
        except exceptions.ServiceUnavailable as e:
            print("Unable to connect to Neo4j:", e)
        if embedding_model is None:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
                query_encode_kwargs={
                    "prompt": "Represent this sentence for searching relevant passages: "
                },
            )
        else:
            self.embedding_model = embedding_model
        self.model = llm_model

    def predict(self, query):
        context = self._rag_retrieval(query)
        prompt = (
            f"**context**:\n{context}\n\n**prompt**:\n{query}" if context else query
        )
        response = self.model.chat(
            # model="ministral-3:3b-cloud",
            model="ministral-3:8b-cloud",
            # model="ministral-3:14b-cloud",
            # model="gpt-oss:20b-cloud",
            # model="gpt-oss:120b-cloud",
            # model="mistral-large-3:675b-cloud",
            messages=[{"role": "user", "content": prompt}],
        )
        return context, response.message.content

    def load(self, llm_model: Client, **kwargs):
        if llm_model:
            self.model = llm_model
            return True
        return False

    def close(self):
        self.check_db()
        self.driver.close()

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
        with self.driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            result = session.run(
                graph_query,
                {"query_embedding": query_embedding, "top_k": top_k},
            )
            context = result.data()
        return context

    def check_db(self):
        if self.driver_is_connected:
            return
        else:
            raise exceptions.ServiceUnavailable(
                "GraphRAG cannot be used without a valid db connection"
            )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    graph_rag = GraphRAG()
    query = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
    print("Query:", query)
    for n_hop in range(2):
        context = graph_rag._rag_retrieval(query, n_hop=n_hop)

        print("n-hops:", n_hop)
        for record in context:
            print(f'score: {record["score"]:.3f} | {record["text"]}')
