import pickle

import gcn_improve.utils as utils


EDGE_PATH = r"../elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"
CLASS_PATH = r"../elliptic_bitcoin_dataset/elliptic_txs_classes.csv"
FEATURE_PATH = r"../elliptic_bitcoin_dataset/elliptic_txs_features.csv"


class Parser(object):
    def __init__(self):
        self.edge_path = EDGE_PATH
        self.class_path = CLASS_PATH
        self.feature_path = FEATURE_PATH

        self.dump_path = "./data.pkl"
        self.load_path = self.dump_path

        self.edges = []
        self.node2label = {}
        self.node2feature = {}

    def parse_data(self):
        # Load data
        print("---------- Start reading csv files... ----------")
        self.load_edge()
        self.load_class()
        self.load_feature()
        print("---------- Reading finish. ----------")
        # Parse data

    def load_edge(self):
        data = utils.load_csv(self.edge_path)
        is_hander_removed = False

        for line in data:
            if not is_hander_removed:
                is_hander_removed = True
                continue
            source = line[0]
            target = line[1]
            self.edges.append((source, target))

    def load_class(self):
        data = utils.load_csv(self.class_path)
        is_hander_removed = False

        for line in data:
            if not is_hander_removed:
                is_hander_removed = True
                continue
            source = line[0]
            label = line[1]
            self.node2label[source] = label

    def load_feature(self):
        data = utils.load_csv(self.feature_path)
        is_hander_removed = False

        for line in data:
            if not is_hander_removed:
                is_hander_removed = True
                continue
            source = line[0]
            features = [float(num) for num in line[1:]]
            self.node2feature[source] = features

    def dump(self):
        with open(self.dump_path, "wb") as file:
            pickle.dump([self.edges, self.node2label, self.node2feature], file)
        print("---------- Successfully save graph. ----------")

    def load(self):
        with open(self.load_path, "rb") as file:
            [self.edges, self.node2label, self.node2feature] = pickle.load(file)
        print("---------- Successfully load graph. ----------")


if __name__ == '__main__':
    parser = Parser()
    parser.parse()
    parser.dump()
    pass
