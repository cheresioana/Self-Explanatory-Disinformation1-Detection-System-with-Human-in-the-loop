import os
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd

from LLM.orchestrator import fetch_embedding
from utils import clean_text


def process_row(title):
    """Cleans the title and gets its embedding."""
    return fetch_embedding(clean_text(title))

def get_embeddings(df):
    print(f"Get embeddings for {df.shape}")
    embs = [None] * len(df)  # Preallocate list for efficiency

    with ThreadPoolExecutor() as executor:
        future_to_idx = {executor.submit(process_row, df.loc[idx, 'statement']): idx for idx in df.index}

        for i, future in enumerate(as_completed(future_to_idx)):
            idx = future_to_idx[future]
            embs[idx] = future.result()  # Store result in correct position

            # Print progress every 100 rows
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(df)} embeddings...")

    df['embedding'] = embs
    return df

if __name__=='__main__':
    file_name = 'final_dataframe.csv'
    df = pd.read_csv(file_name)
    df = get_embeddings(df)
    df.to_csv('final_dataframe_w_emb.csv')
