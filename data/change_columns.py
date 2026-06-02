import pandas as pd

def change_columns_of_original_dataset(path):
    for split in ["train", "val", "test"]:
        file_path = f'{path}/{split}/{split}.parquet'
        df = pd.read_parquet(file_path)
        df = df.rename(columns={'generation': 'text', 'model': 'generated_by'})
        df = df[['text', 'generated_by']]
        df.to_parquet(file_path, index=False)

path = "./Dataset/Dataset_original"
change_columns_of_original_dataset(path)
