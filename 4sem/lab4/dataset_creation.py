import pandas as pd

big_dataset_df = pd.read_excel('../data/big_dataset.xlsx')

medium_dataset_df = big_dataset_df.sample(n=40000, random_state=42).reset_index(drop=True)
small_dataset_df = big_dataset_df.sample(n=5000, random_state=42).reset_index(drop=True)


medium_dataset_df.to_excel('../data/medium_dataset.xlsx', index=False)
small_dataset_df.to_excel('../data/small_dataset.xlsx', index=False)