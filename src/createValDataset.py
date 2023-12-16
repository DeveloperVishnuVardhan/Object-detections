"""
Karan Shah and Jyothi Vishnu
CS-7180 Fall 2023 semester
Splits the data into train and validation
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def createValDataset(trainDataPath):
    """
    Method to split the data into train and val
    """
    # Load the CSV file
    df = pd.read_csv(trainDataPath)

    # Split the data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.20, random_state=42)

    # Save the splits as new CSV files (optional)
    train_df.to_csv('data/trainSet.csv', index=False)
    val_df.to_csv('data/valSet.csv', index=False)



def main():
    trainPath = "data/train.csv"
    createValDataset(trainPath)


if __name__ == "__main__":
    main()

