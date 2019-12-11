import csv


def load_csv(path: str):
    file = open(path, "r", encoding="utf-8")
    return csv.reader(file)


def perform_logistic_repression(train_data: dict, test_data: dict):
    from sklearn.linear_model import LogisticRegression

    train_features, train_labels = train_data['features'], train_data['labels']
    test_features, test_labels = test_data['features'], test_data['labels']

    lr_model = LogisticRegression()
    lr_model.fit(train_features, train_labels)
    predictions = lr_model.predict(test_features)


if __name__ == '__main__':
    pass
