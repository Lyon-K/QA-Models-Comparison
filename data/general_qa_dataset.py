import pandas as pd
from datasets import load_dataset

def load_nq_open(num_samples=50):
    """
    Load the Natural Questions (Open Domain) dataset.
    This tests the model's internal open-domain factual memory (without external context).
    """
    print("Loading General QA: Natural Questions (nq_open)...")
    dataset = load_dataset("nq_open", split=f"validation[:{num_samples}]")
    
    data_list = []
    for item in dataset:
        question = item["question"]
        answers = item["answer"]
        true_answer = answers[0] if len(answers) > 0 else "No Answer"
        
        data_list.append({
            "Domain": "General QA",
            "Dataset": "Natural Questions",
            "Question": question,
            "Context": "N/A (Open Domain)", 
            "True_Answer": true_answer
        })
        
    return pd.DataFrame(data_list)


def load_hotpot_qa(num_samples=50):
    """
    Load the HotpotQA dataset.
    This tests the model's multi-hop reasoning capabilities in complex contexts.
    """
    print("Loading General QA: HotpotQA...")
    dataset = load_dataset("hotpot_qa", "distractor", split=f"validation[:{num_samples}]")
    
    data_list = []
    for item in dataset:
        question = item["question"]
        
        context_list = []
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]
        for title, text_list in zip(titles, sentences):
            combined_text = " ".join(text_list)
            context_list.append(f"[Title: {title}] {combined_text}")
            
        context = " | ".join(context_list)
        true_answer = item["answer"]
        
        data_list.append({
            "Domain": "General QA",
            "Dataset": "HotpotQA",
            "Question": question,
            "Context": context,
            "True_Answer": true_answer
        })
        
    return pd.DataFrame(data_list)


def load_trivia_qa(num_samples=50):
    """
    Load the TriviaQA dataset.
    This tests the model's ability to handle highly challenging geek/trivia questions.
    """
    print("Loading General QA: TriviaQA...")
    dataset = load_dataset("trivia_qa", "rc", split=f"validation[:{num_samples}]")
    
    data_list = []
    for item in dataset:
        question = item["question"]
        
        context_list = []
        if "search_results" in item and "search_context" in item["search_results"]:
            context_list.extend(item["search_results"]["search_context"])
        context = " | ".join(context_list) if context_list else "N/A"
        
        true_answer = item["answer"]["value"]
        
        data_list.append({
            "Domain": "General QA",
            "Dataset": "TriviaQA",
            "Question": question,
            "Context": context,
            "True_Answer": true_answer
        })
        
    return pd.DataFrame(data_list)


def build_general_qa_benchmark(num_samples_per_dataset=5):
    """
    Build the general Q&A evaluation benchmark.
    """
    print("--- Starting General Q&A Benchmark Pipeline ---")
    
    df_nq = load_nq_open(num_samples_per_dataset)
    df_hotpot = load_hotpot_qa(num_samples_per_dataset)
    df_trivia = load_trivia_qa(num_samples_per_dataset)
    
    qa_benchmark_df = pd.concat([df_nq, df_hotpot, df_trivia], ignore_index=True)
    
    print("--- General Q&A Pipeline Completed! ---")
    return qa_benchmark_df


if __name__ == "__main__":
    qa_data = build_general_qa_benchmark(num_samples_per_dataset=2)
    print("\nGeneral Q&A Sample Output Overview:")
    print(qa_data[["Dataset", "Question"]].head(6))