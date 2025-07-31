import pandas as pd
from sklearn.model_selection import train_test_split


def refactor_df(df):
    X = df.drop('label', axis=1)  # All columns except 'label' are features
    y = df['label']  # Your target label

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y,
        test_size=0.3,  # For example, 20% of data for validation
        random_state=42,  # For reproducibility
        stratify=y  # IMPORTANT: Ensures 'label' distribution is preserved
    )

    df = pd.concat([X_train, y_train], axis=1)
    remain_df =  pd.concat([X_validation, y_validation], axis=1)
    return df, remain_df

if __name__=='__main__':
    df_train = pd.read_csv('data/work_data/veridica_train_emb.csv')
    print(f"Train {df_train.shape}")
    df_validation = pd.read_csv('data/work_data/veridica_evaluation_emb.csv')
    print(f"Validation {df_validation.shape}")
    df_test = pd.read_csv('data/work_data/veridica_test_emb.csv')
    print(f"Test {df_test.shape}")
    df_exp = pd.read_csv('data/work_data/veridica_correction_emb.csv')
    print(f"Expert correction {df_exp.shape}")