import pandas as pd

DATA_DIR = 'data/'


def load_data(path, label, *features):
    df = pd.read_csv(path)
    features, labels = df[list(*features)], df[label]
    return features, labels


def fet_lab_names(features, labels):
    assert isinstance(features, pd.DataFrame)
    assert isinstance(labels, pd.Series)
    return list(features.columns), list(map(str, list(labels.unique())))
