from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from ollama import chat
import ast
import spacy

from graphRAG import GraphRAG


# Load environment variables
load_dotenv()

# Load NLP model
nlp = spacy.load("en_core_web_sm")


def build_graph(graph_rag: GraphRAG, dataset):
    for question, _context, answer in dataset.itertuples(index=False):
        print(f"Extracting triplets: {question} ({answer})")
        triplets = extract_triplets(question, answer)
        print(f"Extracted triplets: {triplets}")
        add_triplets_to_neo4j(graph_rag, triplets)


def add_triplets_to_neo4j(graph_rag: GraphRAG, triplets):
    with graph_rag.driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
        for e1, rel, e2 in triplets:
            print(f"Adding triplet: ({e1} -> {rel} -> {e2})")
            session.execute_write(add_triplet, e1, rel, e2)


def clear_db(graph_rag: GraphRAG):
    with graph_rag.driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
        return session.run("MATCH (n) DETACH DELETE n")


def extract_ner(text):
    entities = [(ent.text, ent.label_) for ent in nlp(text).ents]
    entities = "\n".join([f" - ({entity}, {label})" for (entity, label) in entities])
    if entities:
        entities = "Named Entity pairs:\n" + entities

    return f"{text}\n{entities}"


def chunk_text(text, chunk_size=300):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i : i + chunk_size])


def generate_prompt(question, answer, context=None):
    p = f"""Extract entity-relation triplets (Entity1, Relation, Entity2) for a Knowledge Graph from the given Question and Answer. The named entities extracted are provided below each text. Use these entities to identify entity relationships, paying particular attention to dates and temporal entities.

Output Format:
Return the output as a **list** of **tuples** in the following exact format with no line breaks:
[("entity1", "relationship", "entity2"), ("entity1", "relationship", "entity2"), ...]

Rules:
- **Entities** must be strings and **cannot be empty string**.
- **Relationships** must be strings and **cannot be empty string**
- **Do not include empty entities or relationships** in any of the triplets. Each entity and relationship should be meaningful and cannot be an empty string.
- **Do not include invalid triplets** like ("", "", "") or ("Entity", "", "Entity2").

Only return the list of entity-relation triplets in the exact format specified.


Question: '{question}'

Answer: '{answer}'
"""
    if context:
        p += f"\nContext: {context}\n"
    return p


def extract_triplets(question, answer, context=None, logging=False):
    question = extract_ner(question)
    answer = extract_ner(answer)
    prompt = generate_prompt(question=question, answer=answer)

    invalid_response_count = 0
    while invalid_response_count < 3:
        response = chat(
            model="nuextract",
            messages=[{"role": "user", "content": prompt}],
        )
        if logging:
            print("PROMPT:", prompt)
            print("RAW RESPONSE:", response.message.content)
        response = [
            content
            for content in response.message.content.split("<|end-output|>")
            if content.strip()
        ]
        try:
            triplets = [tuple(lit) for lit in ast.literal_eval(response[-1])]
            assert np.all([len(triplet) % 3 == 0 for triplet in triplets])
            break
        except Exception as e:
            invalid_response_count += 1
            print("Error parsing response:", e)
            print("Response content:", response)
            print("Retry Attempt:", invalid_response_count)
    if invalid_response_count == 3:
        print("Failed to extract the following in 3 attempts:")
        print("question:", question)
        print("answer:", answer)
        return []
    return triplets


def add_triplet(tx, e1, rel, e2):
    query = (
        "MERGE (a:Entity {name: $e1})"
        "MERGE (b:Entity {name: $e2})"
        "MERGE (a)-[r:REL {type: $rel}]->(b)"
    )
    tx.run(query, e1=e1, rel=rel, e2=e2).single()


if __name__ == "__main__":
    mock_dataset = pd.read_parquet("../data/sample_data.parquet").drop(
        "Dataset", axis=1
    )

    graph_rag = GraphRAG()
    # clear_db(graph_rag)
    graph_rag.build_graph(mock_dataset)

# For chunking context(currently unusable)
# import tiktoken

# encoding = tiktoken.get_encoding("cl100k_base")

# def tokenizer(text):
#     return len(encoding.encode(text))

# def chunk_context_for_prompt(prompt, context, question, answer, max_tokens=4096, tokenizer=tokenizer):
#     """
#     context: str
#     question: str
#     answer: str
#     max_tokens: int, model max tokens
#     tokenizer: function to get number of tokens
#     """
#     question = extract_NER(question)
#     answer = extract_NER(answer)
# Context is unusable: title & sentences are merged(https://huggingface.co/datasets/hotpotqa/hotpot_qa/viewer/distractor/train?row=4)
# context_chunks = []
# current_chunk = ""

# # compute available space for context
# fixed_tokens = tokenizer(prompt(question=question, answer=answer, context=""))
# available_tokens = max_tokens - fixed_tokens

# for sentence in sentences:
#     if tokenizer(current_chunk + sentence) <= available_tokens:
#         current_chunk += sentence + ". "
#     else:
#         context_chunks.append(current_chunk.strip())
#         current_chunk = sentence + ". "

# if current_chunk:
#     context_chunks.append(current_chunk.strip())

# return context_chunks

# # Generate full prompts
# prompts = [prompt(question, c, answer) for c in chunk_context_for_prompt(context, question, answer, 4096, tokenizer)]
