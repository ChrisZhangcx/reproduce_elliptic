import csv


def load_csv(path: str):
    file = open(path, "r", encoding="utf-8")
    return csv.reader(file)


if __name__ == '__main__':
    pass
