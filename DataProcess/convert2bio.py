import pandas as pd
import utils.path as path


def read_csv(path):
    df = pd.read_csv(path)
    df = df.drop('PATIENTSITE', 1)
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def add_fullstop(df):
    for index, row in df.iterrows():
        if df.iloc[index, df.columns.get_loc("CHECKDIAGOPIN")][-1] != '。':
            if df.iloc[index, df.columns.get_loc("CHECKDIAGOPIN")][-1] != '，':
                df.iloc[index, df.columns.get_loc("CHECKDIAGOPIN")] += '。'
            # else:
            #     df.iloc[index, df.columns.get_loc("CHECKDIAGOPIN")][-1] = '。'
    return df


def export(df):
    with open(path.unlabeled_bio, 'w+') as f:
        for index, row in df.iterrows():
            for chars in row['export']:
                f.write(chars + '\tO\n')
            f.write("\n")


if __name__ == '__main__':
    original_df = read_csv(path.origincal_csv_data)
    added_fullstop_df = add_fullstop(original_df)
    added_fullstop_df["export"] = added_fullstop_df["CHECKSEERECORD"].map(str) + added_fullstop_df["CHECKDIAGOPIN"]
    print(added_fullstop_df.head())
    export(added_fullstop_df)
