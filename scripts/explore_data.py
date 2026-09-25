import pandas as pd

df = pd.read_parquet("data/arena_55k.parquet")
print(df.head())
# print("Printing the columns")

print(df.iloc[0])

df2 = pd.read_parquet("data/gpt4_judge_train.parquet")
print(df2.iloc[0])
