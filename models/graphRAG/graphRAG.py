import os
import re
from typing import Literal, Optional

from neo4j import Driver, GraphDatabase, exceptions
from langchain_huggingface import HuggingFaceEmbeddings
from ollama import Client
import logging

try:
    import certifi
except Exception:  # pragma: no cover - optional dependency
    certifi = None

logger = logging.getLogger(__name__)

GRAPH_RELATION_PATTERN = re.compile(
    r"^(?P<src_label>[A-Z_]+)\s+(?P<src_name>.+?)\s+(?P<relation>[A-Z_]+)\s+(?P<tgt_label>[A-Z_]+)\s+(?P<tgt_name>.+)$"
)


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
        self._DATABASE = os.getenv("NEO4J_DATABASE", "2cb6a311")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        if not all([self._URI, user, password]):
            raise ValueError("Neo4j credentials missing")
        self._AUTH = (user, password)

        if certifi is not None and not os.getenv("SSL_CERT_FILE"):
            os.environ["SSL_CERT_FILE"] = certifi.where()

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

    def _format_graph_context(self, context_records: list[dict]) -> str:
        if not context_records:
            return "Graph evidence is limited for this query."

        evidence_lines: list[str] = []
        entities: list[str] = []
        relations: list[str] = []

        for record in context_records[:4]:
            raw_text = str(record.get("text", "")).strip()
            if not raw_text:
                continue

            match = GRAPH_RELATION_PATTERN.match(raw_text)
            if match:
                source = f"{match.group('src_name')} [{match.group('src_label')}]"
                relation = match.group("relation")
                target = f"{match.group('tgt_name')} [{match.group('tgt_label')}]"
                evidence_lines.append(
                    f"Entity: {source}\nRelation: {relation}\nEntity: {target}"
                )
                entities.extend([match.group("src_name"), match.group("tgt_name")])
                relations.append(relation)
            else:
                evidence_lines.append(f"Graph snippet: {raw_text}")

        unique_entities: list[str] = []
        seen_entities: set[str] = set()
        for entity in entities:
            key = entity.strip().lower()
            if key and key not in seen_entities:
                seen_entities.add(key)
                unique_entities.append(entity.strip())

        unique_relations: list[str] = []
        seen_relations: set[str] = set()
        for relation in relations:
            key = relation.strip().lower()
            if key and key not in seen_relations:
                seen_relations.add(key)
                unique_relations.append(relation.strip())

        sections = []
        if unique_entities:
            sections.append(
                "Relevant graph entities include: " + ", ".join(unique_entities[:6])
            )
        if unique_relations:
            sections.append(
                "Observed graph relations: " + ", ".join(unique_relations[:6])
            )
        sections.append("Retrieved graph evidence:\n" + "\n\n".join(evidence_lines[:4]))
        return "\n\n".join(sections)

    def predict(self, query, n_hop: Literal[0, 1] = 0):
        context = self._rag_retrieval(query, n_hop=n_hop)
        graph_context = self._format_graph_context(context)
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

        prompt = f"""Use the retrieved graph context to answer the user query.

Focus on graph-grounded evidence rather than generic background knowledge.
If the graph context is strong, explicitly describe how entities are connected and what the relations suggest.
If the graph context is limited, say that neutrally and still answer as best as possible from the retrieved graph evidence.
Prefer concise, relation-aware wording such as:
- "Based on the retrieved graph context..."
- "The graph indicates..."
- "Relevant connected entities include..."
- "The relationship structure suggests..."

Retrieved graph context:
{graph_context}

User query:
{query}
"""
        response = self.llm_model.chat(
            model="ministral-3:8b-cloud",
            messages=[{"role": "user", "content": prompt}],
        )
        return context, response.message.content

    def close(self):
        if hasattr(self, "driver") and self.driver is not None:
            self.driver.close()
            self.driver = None

    def _rag_retrieval(self, query, n_hop=1, top_k=3):
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
        with self.driver.session(database=self._DATABASE) as session:
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
