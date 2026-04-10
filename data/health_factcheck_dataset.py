import pandas as pd
from datasets import load_dataset

def load_pubhealth(num_samples=50):
    """
    Load the PUBHEALTH (health_fact) dataset.
    Focuses on fact-checking public health news claims.
    """
    print("Loading Health Fact-Check: PUBHEALTH...")
    dataset = load_dataset("health_fact", split=f"validation[:{num_samples}]")
    
    # Label mapping dictionary
    label_map = {0: "false", 1: "mixture", 2: "true", 3: "unproven", -1: "unknown"}
    
    data_list = []
    for item in dataset:
        claim_to_verify = item["claim"]
        background_text = item["main_text"] if item["main_text"] else "N/A"
        
        label_int = item["label"]
        label_str = label_map.get(label_int, "unknown")
        explanation = item["explanation"] if item["explanation"] else "No explanation provided."
        
        verdict_result = f"[{label_str.upper()}] {explanation}"
        
        data_list.append({
            "Domain": "Public Health",
            "Dataset": "PUBHEALTH",
            "Claim": claim_to_verify,
            "Evidence_Context": background_text,
            "Target_Verdict": verdict_result
        })
        
    return pd.DataFrame(data_list)


def load_healthver(num_samples=50):
    """
    Load the HealthVer dataset.
    Focuses on verifying medical claims against scientific literature abstracts 
    (highly suitable for RAG scenarios).
    """
    print("Loading Health Fact-Check: HealthVer...")
    dataset = load_dataset("dwadden/healthver_entailment", split=f"train[:{num_samples}]")
    
    data_list = []
    for item in dataset:
        claim_to_verify = item["claim"]
        
        abstract_sentences = item["abstract"]
        scientific_context = " ".join(abstract_sentences) if abstract_sentences else "N/A"
        
        verdict = item["verdict"] if item["verdict"] else "Unknown"
        evidence_list = item["evidence"]
        evidence_text = " ".join(evidence_list) if evidence_list else "No evidence provided."
        
        verdict_result = f"[{verdict.upper()}] {evidence_text}"
        
        data_list.append({
            "Domain": "Public Health",
            "Dataset": "HealthVer",
            "Claim": claim_to_verify,
            "Evidence_Context": scientific_context,
            "Target_Verdict": verdict_result
        })
        
    return pd.DataFrame(data_list)


def build_health_factcheck_benchmark(num_samples_per_dataset=5):
    """
    Build the public health misinformation fact-checking evaluation benchmark.
    """
    print("--- Starting Public Health Fact-Checking Pipeline ---")
    
    df_pubhealth = load_pubhealth(num_samples_per_dataset)
    df_healthver = load_healthver(num_samples_per_dataset)
    
    health_benchmark_df = pd.concat([df_pubhealth, df_healthver], ignore_index=True)
    
    print("--- Public Health Pipeline Completed! ---")
    return health_benchmark_df


if __name__ == "__main__":
    health_data = build_health_factcheck_benchmark(num_samples_per_dataset=2)
    
    print("\nPublic Health Sample Output Overview:")
    # Print the first few rows to view the output. 
    # Field names are 'Claim' and 'Target_Verdict' to better fit the fact-checking context.
    print(health_data[["Dataset", "Claim", "Target_Verdict"]].head(4))