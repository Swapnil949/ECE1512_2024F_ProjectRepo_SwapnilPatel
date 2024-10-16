import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def get_split_data() -> pd.DataFrame:
    df = pd.read_csv("annotations.csv")
    df_train = df.query('Partition == "train"')
    df_test = df.query('Partition == "test"')

    return df_train, df_test

def main():
    split_root = "images-split"
    Path(split_root).mkdir(exist_ok=True)

    def make_symlink(row):
        # get current directory
        src = Path.cwd() / Path("images") / row["Image Name"]
        label = row["Majority Vote Label"]
        partition = row["Partition"]
        dst = Path(split_root) / partition / label / row["Image Name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src, dst)

    df_train, df_test = get_split_data()
    df_train.to_csv("train_annotations.csv", index=False)
    df_test.to_csv("test_annotations.csv", index=False)

    df_train.apply(make_symlink, axis=1)
    df_test.apply(make_symlink, axis=1)

if __name__ == "__main__":
    main()
