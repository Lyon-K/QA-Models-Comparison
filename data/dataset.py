import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def data_preparation(num_samples_per_dataset=20):
    """
    Loads both Public Health Fact-Checking and General QA datasets,
    aligns their column names, and splits them into mixed training and testing sets.
    """
    logger.info("Loading BOTH Public Health and General QA datasets...")
    
    # Import the data generation scripts
    from data.health_factcheck_dataset import build_health_factcheck_benchmark
    from data.general_qa_dataset import build_general_qa_benchmark
    
    df_health = build_health_factcheck_benchmark(num_samples_per_dataset)
    df_qa = build_general_qa_benchmark(num_samples_per_dataset)
    
    # Data Alignment: Map to Input_Query, Input_Context, Target_Output
    df_health_aligned = df_health.rename(columns={
        "Claim": "Input_Query",
        "Evidence_Context": "Input_Context",
        "Target_Verdict": "Target_Output"
    })
    
    df_qa_aligned = df_qa.rename(columns={
        "Question": "Input_Query",
        "Context": "Input_Context",
        "True_Answer": "Target_Output"
    })
    
    # Concatenate into a single unified dataframe
    unified_df = pd.concat([df_health_aligned, df_qa_aligned], ignore_index=True)
    
    # Extract features (X) and labels (y)
    X = unified_df[["Input_Query", "Input_Context", "Domain"]].to_dict(orient="records")
    y = unified_df["Target_Output"].tolist()
    
    # Shuffle and split the dataset (80% train, 20% test)
    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    logger.info(f"Mixed data split complete: {len(train_x)} training samples, {len(test_x)} testing samples.")
    
    if len(test_x) > 0:
        logger.info(f"Sample X format: {test_x[0]}")
        
    return train_x, test_x, train_y, test_y