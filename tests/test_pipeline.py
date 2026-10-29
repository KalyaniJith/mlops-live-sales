import os
import pandas as pd

def test_dataset_exists():
    assert os.path.exists("data/live.csv")

def test_dataset_has_required_columns():
    df = pd.read_csv("data/live.csv")
    required = ["status_type", "num_comments", "num_shares", "num_reactions"]
    assert all(col in df.columns for col in required)

def test_dataset_has_no_missing_required_values():
    df = pd.read_csv("data/live.csv")
    required = ["status_type", "num_comments", "num_shares", "num_reactions"]
    assert not df[required].isnull().any().any()

def test_reactions_are_non_negative():
    df = pd.read_csv("data/live.csv")
    assert (df["num_reactions"] >= 0).all()
