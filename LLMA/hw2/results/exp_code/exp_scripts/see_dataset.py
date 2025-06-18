import pandas as pd


train_df = pd.read_parquet("/data/align_anything_t2t/train_30k.parquet")
val_df = pd.read_parquet("/data/align_anything_t2t/val_1k.parquet")


print("Train Data:")
print(train_df.head())

print("\nValidation Data:")
print(val_df.head())


print("\nTrain Data Columns:", train_df.columns)
print("Validation Data Columns:", val_df.columns)
