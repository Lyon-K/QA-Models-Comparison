# DO NOT IMPORT THIS FILE (DEPENDENCY HEAVY)
import os
from ollama import chat
import json
import re
import pandas as pd
from dotenv import load_dotenv

from models.graphRAG.graphRAG import GraphRAG

load_dotenv()


def build_graph(graph_rag: GraphRAG, dataset, logging=False):
    print("Building graph from dataset...")
    driver = graph_rag.driver
    embedding_model = graph_rag.embedding_model
    # chunk_context

    for idx, row in dataset.iterrows():
        try:
            text = f"Question: {row['question']}\nAnswer: {row['answer']}\nContext: {row['context']}"
            print(f"\nInserting into KG: (doc_{idx})\n{text}")
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
    input_format = """Extract knowledge graph triples from the text.

**RULES:**
- Use ONLY allowed predicates.
- Extract concise canonical entities (no full sentences).
- Do NOT create duplicate triples.
- Do NOT create self-loops (A, relation, A).
- Do NOT invent predicates.
- Extract only meaningful factual or clearly implied relations.
- Use ASSOCIATED_WITH only if no stronger predicate fits.

**ENTITY TYPES (for guidance only):**
{entity_types}

**Note:**
- Do NOT output entities separately.
- Ignore QUESTION/ANSWER/TOPIC structure.

**PREDICATES (STRICT):**
{predicates}

**TEXT:**
{text}
"""
    entity_types = [
        "DISEASE",
        "SYMPTOM",
        "DRUG",
        "TREATMENT",
        "PROCEDURE",
        "TEST",
        "PATHOGEN",
        "GENE_OR_PROTEIN",
        "BIOLOGICAL_PROCESS",
        "ANATOMY",
        "RISK_FACTOR",
        "POPULATION",
        "CHEMICAL",
        "CONDITION",
    ]
    predicates = [
        "CAUSES",
        "TREATS",
        "PREVENTS",
        "DIAGNOSES",
        "INDICATES",
        "AFFECTS",
        "INTERACTS_WITH",
        "PART_OF",
        "PRODUCES",
        "INVOLVES",
        "INCREASES_RISK_OF",
        "DECREASES_RISK_OF",
        "PREDISPOSES_TO",
        "TARGETS",
        "HAS_SIDE_EFFECT",
        "AFFECTS_POPULATION",
        "MORE_COMMON_IN",
        "IS_A",
        "ASSOCIATED_WITH",
        "RELATED_TO",
    ]

    prompt = input_format.format(
        entity_types=json.dumps({"entity_types": entity_types}),
        predicates=json.dumps({"predicates": predicates}),
        text=text,
    )

    response = chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "top_p": 1, "num_predict": 2048},
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
    with graph_rag.driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
        return session.run("MATCH (n) DETACH DELETE n")


if __name__ == "__main__":
    import numpy as np
    from data.LYS_dataset import get_dataset

    train, test = get_dataset()
    train["context"] = train["context"].mask(train["source_dataset"] == "MedQuad", "")

    graph_rag = GraphRAG()

    parsed = [
        f"Question: {q}\nAnswer: {a}\nContext: {c}"
        for q, a, c in zip(train["question"], train["answer"], train["context"])
    ]
    # print(len(parsed))
    embeddings = [graph_rag.embedding_model.embed_query(text) for text in parsed]
    # print(len(embeddings))

    from sklearn.cluster import KMeans

    k = 15  # try different values
    kmeans = KMeans(n_clusters=k, random_state=42)
    train["cluster"] = kmeans.fit_predict(embeddings)

    chosen_topic_1 = np.argsort(np.unique_counts(train["cluster"]).counts)[-1]
    chosen_topic_2 = np.argsort(np.unique_counts(train["cluster"]).counts)[-3]
    target_df = pd.concat(
        [
            train[train["cluster"] == chosen_topic_1],
            train[train["cluster"] == chosen_topic_2],
        ],
        ignore_index=True,
    )

    target_df["context"] = ""
    target_df["chunked_context"] = ""
    target_df.to_csv("graphrag_target_data.csv")
    # print(target_df)

    clear_db(graph_rag)
    # print("Building graph")
    build_graph(graph_rag, target_df, logging=True)
