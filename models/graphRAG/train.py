# DO NOT IMPORT THIS FILE (DEPENDENCY HEAVY)
import os
from ollama import chat
import json
import re
import pandas as pd
from dotenv import load_dotenv

from graphRAG import GraphRAG

load_dotenv()


def build_graph(graph_rag: GraphRAG, dataset, logging=False):
    print("Building graph from dataset...")
    driver = graph_rag.driver
    embedding_model = graph_rag.embedding_model
    for question, _context, answer in dataset.itertuples(index=False):
        try:
            print(f"\nInserting into KG: {question} ({answer})")
            text = f"Question: {question}\nAnswer: {answer}"
            triplets = triplextract_ollama(text)
            if logging:
                print(f"Extracted triplets: {triplets}")

            if driver:
                insert_neo4j(driver, triplets, embedding_model, logging=logging)
            else:
                print("No Neo4j driver available. Skipping database insertion.")
                if logging:
                    print("Triplets for {text}:\n{triplets}")
        except Exception as e:
            print("Error:", e)


def triplextract_ollama(text, model_name="sciphi/triplex"):
    input_format = """Perform Named Entity Recognition (NER) and extract knowledge graph triplets from the text. 
NER identifies named entities of given entity types, and triple extraction identifies relationships between entities using specified predicates.

**Entity Types:**
{entity_types}

**Predicates:**
{predicates}

**Text:**
{text}
"""
    entity_types = [
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "DATE",
        "EVENT",
        "WORK",
        "NUMBER",
    ]
    predicates = [
        "BORN_IN",
        "DIED_IN",
        "LOCATED_IN",
        "WORKED_AT",
        "FOUNDED",
        "CREATED",
        "PART_OF",
        # "HAS_ATTRIBUTE",
        "DATE_OF",
        "CAUSE_OF",
        "RELATED_TO",
        "ANSWER_TO",
    ]

    prompt = input_format.format(
        entity_types=json.dumps({"entity_types": entity_types}),
        predicates=json.dumps({"predicates": predicates}),
        text=text,
    )

    response = chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "top_p": 1, "num_predict": 512},
    )

    return response["message"]["content"]


def triplets_to_cypher_s(triplet_str, embedding_model):
    if triplet_str.startswith("```") and triplet_str.endswith("```"):
        triplet_str = "\n".join(triplet_str.split("\n")[1:-1])

    triplet_json = json.loads(triplet_str)
    cypher_cmds = []
    relations = []
    literals = []
    node_info = {}

    for item in triplet_json["entities_and_triples"]:
        # ENTITY
        m = re.match(r"\[(\d+)\],\s*(\w+):(.+)", item)
        if m:
            idx, etype, name = m.groups()
            label = re.sub(r"[^A-Z0-9_]", "_", etype.upper())
            node_info[idx] = {"name": name.strip(), "type": label}
            embedding_text = f"{label} {name.strip()}"
            embedding = embedding_model.embed_query(embedding_text)
            cypher_cmds.append(
                f'MERGE (n{idx}:{label} {{name: "{name.strip()}", embedding_text: "{embedding_text}", embedding:{embedding}}})'
            )
            continue

        # RELATION: [1] REL [2]
        m = re.match(r"\[(\d+)\]\s+(\w+)\s+\[(\d+)\]", item)
        if m:
            relations.append(m.groups())
            continue

        # RELATION with literal: [1] REL value
        m = re.match(r"\[(\d+)\]\s+(\w+)\s+(.+)", item)
        if m:
            literals.append(m.groups())

    for src, rel, tgt in relations:
        # Generate embedding_text for triple
        src_node = node_info[src]
        tgt_node = node_info[tgt]
        embedding_text = f"{src_node['type']} {src_node['name']} {rel} {tgt_node['type']} {tgt_node['name']}"
        embedding = embedding_model.embed_query(embedding_text)
        # Merge triple as relationship
        cypher_cmds.append(
            f'MERGE (n{src})-[:{rel} {{embedding_text: "{embedding_text}", embedding:{embedding}}}]->(n{tgt})'
        )

    for src, rel, value in literals:
        src_node = node_info[src]
        embedding_text = f"{src_node['type']} {src_node['name']} {rel} {value.strip()}"
        embedding = embedding_model.embed_query(embedding_text)
        cypher_cmds.append(
            f'MERGE (n{src})-[:{rel} {{embedding_text: "{embedding_text}", embedding:{embedding}}}]->(:Literal {{value: "{value.strip()}"}})'
        )

    return cypher_cmds


def insert_neo4j(driver, triplet_str, embedding_model, logging=False):
    cypher_command = "\n".join(triplets_to_cypher_s(triplet_str, embedding_model))
    if logging:
        print("Executing cypher command:")
        print(cypher_command)

    with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
        session.run(cypher_command)


def clear_db(graph_rag: GraphRAG):
    db = os.getenv("NEO4J_DATABASE")
    print("Clearing Neo4j database:", db)
    with graph_rag.driver.session(database=db) as session:
        return session.run("MATCH (n) DETACH DELETE n")


if __name__ == "__main__":
    graph_rag = GraphRAG()
    clear_db(graph_rag)

    mock_dataset = pd.read_parquet("data/sample_data.parquet").drop("Dataset", axis=1)
    build_graph(graph_rag, mock_dataset, logging=False)
