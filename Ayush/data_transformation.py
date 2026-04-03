import pandas as pd
df = pd.read_parquet('train.parquet', engine='fastparquet')

#######_______Remove Outliers_______#######
def remove_outliers(df):
  word_counts = df['text'].apply(lambda x: len(x.split()))
  Q1 = word_counts.quantile(0.25)
  Q3 = word_counts.quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df_removed_outliers = df[(word_counts >= lower_bound) & (word_counts <= upper_bound)].copy()
  return df_removed_outliers


#######_______max word count per data = 500_______#######

# Truncate the data
def truncate(df, max_words = 500):
  df_truncated = df.copy()
  df_truncated['text'] = (df_truncated['text'].str.split().str[:max_words].str.join(" "))
  return df_truncated

# Split the data
def split_texts(df, max_length=500):
    df_split = df.copy()
    def chunk_words(text):
        words = text.split()
        return [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    df_split['text'] = df_split['text'].apply(chunk_words)
    df_exploded = df_split.explode('text', ignore_index=True)
    return df_exploded


