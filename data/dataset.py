import pandas as pd


def load_dataset():
    # Example placeholder data
    df = pd.read_parquet("data/sample_data.parquet")
    y_column = "True_Answer"

    train_x = df.drop(y_column, axis=1)
    train_y = df[y_column]
    test_x = None
    test_y = None

    return train_x, test_x, train_y, test_y
