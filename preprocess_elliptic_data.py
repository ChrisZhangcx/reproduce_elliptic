import os
import pickle

import utils as utils


EDGE_PATH = r"../data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"
CLASS_PATH = r"../data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv"
FEATURE_PATH = r"../data/elliptic_bitcoin_dataset/elliptic_txs_features.csv"


class Parser(object):
    def __init__(self):
        self.edge_path = EDGE_PATH
        self.class_path = CLASS_PATH
        self.feature_path = FEATURE_PATH

        self.dump_path = r"../data/elliptic_bitcoin_dataset"
        self.data_path = os.path.join(self.dump_path, "data.pkl")
        self.load_path = self.dump_path

        self.edges = []
        self.node2label = {}
        self.node2feature = {}

    def parse_data(self, is_dump: bool = False):
        # Load data
        if os.path.exists(self.data_path):
            print("---------- Loading pre-parsed data... ----------")
            self.load()
        else:
            print("---------- Start reading csv files... ----------")
            self.load_edge()
            self.load_class()
            self.load_feature()
            self.dump()
        print("---------- Data pre-processing finish. ----------")
        self.statistics()

        # 136265

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

        for line in data:
            source = line[0]
            features = [float(num) for num in line[1:]]
            self.node2feature[source] = features

    def statistics(self):
        if len(self.edges) == 0:
            print("---------- Data have not been initialized yet. ----------")
            return

        # general
        print("\n---------- Data Info ----------")
        print(f"----- Nodes: {len(self.node2label.keys())} -----")
        print(f"----- Edges: {len(self.edges)} -----")

        # label related
        labelset = set(self.node2label.values())
        label2num = {label: 0 for label in labelset}
        for l in self.node2label.values():
            label2num[l] += 1
        print(f"----- {label2num} -----")

        # correference related
        feature_nodes = set(self.node2feature.keys())
        label_nodes = set(self.node2label.keys())
        diff1 = feature_nodes - label_nodes
        diff2 = label_nodes - feature_nodes
        print(f"Node differences: {diff1}, {diff2}")

        # feature related
        for key in self.node2feature.keys():
            print(f"----- Feature len: {len(self.node2feature[key])} -----")
            break

    def dump(self):
        with open(self.data_path, "wb") as file:
            pickle.dump([self.edges, self.node2label, self.node2feature], file)
        print("---------- Successfully save graph. ----------")

    def load(self):
        with open(self.data_path, "rb") as file:
            [self.edges, self.node2label, self.node2feature] = pickle.load(file)
        print("---------- Successfully load graph. ----------")

    def parse_to_gcn_dataset(self):
        if len(self.edges) == 0:
            print("---------- Data have not been initialized yet. ----------")
            return

        print("\n---------- Generating gcn-usage data format... ----------")
        # elliptic.cites
        edges_path = os.path.join(self.dump_path, "elliptic.cites")
        with open(edges_path, "w", encoding="utf-8") as file:
            for e in self.edges:
                file.write(f"{e[0]} {e[1]}\n")

        # elliptic.content
        content_path = os.path.join(self.dump_path, "elliptic.content")
        with open(content_path, "w", encoding="utf-8") as file:
            for node in self.node2label:
                label = self.node2label[node]
                feature = self.node2feature[node]
                file.write(f"{node} {' '.join([str(f) for f in feature])} {label}\n")


if __name__ == '__main__':
    parser = Parser()
    parser.parse_data(is_dump=False)

    parser.parse_to_gcn_dataset()
