from dotenv import load_dotenv
import os
from neo4j import GraphDatabase

load_dotenv()


class GraphRAG:
    def __init__(self):
        print("Building graph from dataset...")
        self.driver = GraphDatabase.driver(
            uri=os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
        )
