import numpy as np
import os
from collections import defaultdict
import json
import dgl
import torch
from pprint import pprint


def load_hexaruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        hexapleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            att_head1 = float(line_split[3])
            att_tail1 = float(line_split[4])
            att_head2 = float(line_split[5])
            att_tail2 = float(line_split[6])
            time = int(line_split[7])
            hexapleList.append([
                head, rel, tail, att_head1, att_tail1, att_head2, att_tail2,
                time
            ])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                att_head1 = float(line_split[3])
                att_tail1 = float(line_split[4])
                att_head2 = float(line_split[5])
                att_tail2 = float(line_split[6])
                time = int(line_split[7])
                hexapleList.append([
                    head, rel, tail, att_head1, att_tail1, att_head2,
                    att_tail2, time
                ])
                times.add(time)
    times = list(times)
    times.sort()

    return hexapleList, times


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return triples


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm


train_data, train_times = load_hexaruples('', 'train.txt')
test_data, test_times = load_hexaruples('', 'test.txt')
valid_data, valid_times = load_hexaruples('', 'valid.txt')
# total_data, _ = load_hexaruples('', 'train.txt', 'test.txt')

history_len = 10
num_e, num_r = get_total_number('', 'stat.txt')

entity_s_his = [[] for _ in range(num_e)]
att_s_his = [[] for _ in range(num_e)]
rel_s_his = [[] for _ in range(num_e)]
self_att_s_his = [[] for _ in range(num_e)]

entity_o_his = [[] for _ in range(num_e)]
att_o_his = [[] for _ in range(num_e)]
rel_o_his = [[] for _ in range(num_e)]
self_att_o_his = [[] for _ in range(num_e)]

# s_his_t = [[] for _ in range(num_e)]
# o_his_t = [[] for _ in range(num_e)]

entity_s_history_data = [[] for _ in range(len(train_data))]
att_s_history_data = [[] for _ in range(len(train_data))]
rel_s_history_data = [[] for _ in range(len(train_data))]
self_att_s_history_data = [[] for _ in range(len(train_data))]

entity_o_history_data = [[] for _ in range(len(train_data))]
att_o_history_data = [[] for _ in range(len(train_data))]
rel_o_history_data = [[] for _ in range(len(train_data))]
self_att_o_history_data = [[] for _ in range(len(train_data))]

e = []
r = []

latest_t = 0

entity_s_his_cache = [[] for _ in range(num_e)]
att_s_his_cache = [[] for _ in range(num_e)]
rel_s_his_cache = [[] for _ in range(num_e)]
self_att_s_his_cache = [[] for _ in range(num_e)]

entity_o_his_cache = [[] for _ in range(num_e)]
att_o_his_cache = [[] for _ in range(num_e)]
rel_o_his_cache = [[] for _ in range(num_e)]
self_att_o_his_cache = [[] for _ in range(num_e)]

for i, train in enumerate(train_data):
    if i % 10000 == 0:
        print("train", i, len(train_data))
    # if i == 10000:
    #     break
    t = int(train[-1])
    if latest_t != t:

        for ee in range(num_e):
            if len(entity_s_his_cache[ee]) != 0:
                if len(entity_s_his[ee]) >= history_len:
                    entity_s_his[ee].pop(0)
                    att_s_his[ee].pop(0)
                    rel_s_his[ee].pop(0)
                    self_att_s_his[ee].pop(0)

                entity_s_his[ee].append(entity_s_his_cache[ee].copy())
                att_s_his[ee].append(att_s_his_cache[ee].copy())
                rel_s_his[ee].append(rel_s_his_cache[ee].copy())
                self_att_s_his[ee].append(self_att_s_his_cache[ee].copy())

                entity_s_his_cache[ee] = []
                att_s_his_cache[ee] = []
                rel_s_his_cache[ee] = []
                self_att_s_his_cache[ee] = []

            if len(entity_o_his_cache[ee]) != 0:
                if len(entity_o_his[ee]) >= history_len:
                    entity_o_his[ee].pop(0)
                    att_o_his[ee].pop(0)
                    rel_o_his[ee].pop(0)
                    self_att_o_his[ee].pop(0)

                entity_o_his[ee].append(entity_o_his_cache[ee].copy())
                att_o_his[ee].append(att_o_his_cache[ee].copy())
                rel_o_his[ee].append(rel_o_his_cache[ee].copy())
                self_att_o_his[ee].append(self_att_o_his_cache[ee].copy())

                entity_o_his_cache[ee] = []
                att_o_his_cache[ee] = []
                rel_o_his_cache[ee] = []
                self_att_o_his_cache[ee] = []

        latest_t = t
    s = int(train[0])
    r = int(train[1])
    o = int(train[2])
    att_s = [train[3], train[5]]
    att_o = [train[4], train[6]]
    # print(s_his[r][s])
    entity_s_history_data[i] = entity_s_his[s].copy()
    att_s_history_data[i] = att_s_his[s].copy()
    rel_s_history_data[i] = rel_s_his[s].copy()
    self_att_s_history_data[i] = self_att_s_his[s].copy()

    entity_o_history_data[i] = entity_o_his[o].copy()
    att_o_history_data[i] = att_o_his[o].copy()
    rel_o_history_data[i] = rel_o_his[o].copy()
    self_att_o_history_data[i] = self_att_o_his[o].copy()
    # print(o_history_data_g[i])

    if len(entity_s_his_cache[s]) == 0:
        entity_s_his_cache[s] = [o]
        rel_s_his_cache[s] = [r]
        att_s_his_cache[s] = [att_o]
        self_att_s_his_cache[s] = [att_s]
    else:
        entity_s_his_cache[s] = entity_s_his_cache[s] + [o]
        rel_s_his_cache[s] = rel_s_his_cache[s] + [r]
        att_s_his_cache[s] = att_s_his_cache[s] + [att_o]
        self_att_s_his_cache[s] = self_att_s_his_cache[s] + [att_s]

    if len(entity_o_his_cache[o]) == 0:
        entity_o_his_cache[o] = [s]
        rel_o_his_cache[o] = [r]
        att_o_his_cache[o] = [att_s]
        self_att_o_his_cache[o] = [att_o]
    else:
        entity_o_his_cache[o] = entity_o_his_cache[o] + [s]
        rel_o_his_cache[o] = rel_o_his_cache[o] + [r]
        att_o_his_cache[o] = att_o_his_cache[o] + [att_s]
        self_att_o_his_cache[o] = self_att_o_his_cache[o] + [att_o]

    # print(s_history_data[i], s_history_data_g[i])
    # with open('ttt.txt', 'w') as fp:
    #     json.dump(s_history_data_g, fp, indent=4)
    # print("save")

# with open('train_graphs.txt', 'w') as fp:
#     json.dump(graph_dict_train, fp, indent=4)

# entity_s_history_data = [[list(c) for c in k] for k in entity_s_history_data]
# rel_s_history_data = [[list(c) for c in k] for k in rel_s_history_data]
# att_s_history_data = [[list(c) for c in k] for k in att_s_history_data]
self_att_s_history_data = [[c[0] for c in k] for k in self_att_s_history_data]

# entity_o_history_data = [[list(c) for c in k] for k in entity_o_history_data]
# rel_o_history_data = [[list(c) for c in k] for k in rel_o_history_data]
# att_o_history_data = [[list(c) for c in k] for k in att_o_history_data]
self_att_o_history_data = [[c[0] for c in k] for k in self_att_o_history_data]

# pprint(entity_s_history_data)
# pprint(rel_s_history_data)
# pprint(att_s_history_data[-1])
# pprint(self_att_s_history_data)

with open('train_entity_s_history_data.json', 'w') as fp:
    json.dump(entity_s_history_data, fp, indent=4)
with open('train_rel_s_history_data.json', 'w') as fp:
    json.dump(rel_s_history_data, fp, indent=4)
with open('train_att_s_history_data.json', 'w') as fp:
    json.dump(att_s_history_data, fp, indent=4)
with open('train_self_att_s_history_data.json', 'w') as fp:
    json.dump(self_att_s_history_data, fp, indent=4)

with open('train_entity_o_history_data.json', 'w') as fp:
    json.dump(entity_o_history_data, fp, indent=4)
with open('train_rel_o_history_data.json', 'w') as fp:
    json.dump(rel_o_history_data, fp, indent=4)
with open('train_att_o_history_data.json', 'w') as fp:
    json.dump(att_o_history_data, fp, indent=4)
with open('train_self_att_o_history_data.json', 'w') as fp:
    json.dump(self_att_o_history_data, fp, indent=4)

entity_s_history_data_valid = [[] for _ in range(len(valid_data))]
att_s_history_data_valid = [[] for _ in range(len(valid_data))]
rel_s_history_data_valid = [[] for _ in range(len(valid_data))]
self_att_s_history_data_valid = [[] for _ in range(len(valid_data))]

entity_o_history_data_valid = [[] for _ in range(len(valid_data))]
att_o_history_data_valid = [[] for _ in range(len(valid_data))]
rel_o_history_data_valid = [[] for _ in range(len(valid_data))]
self_att_o_history_data_valid = [[] for _ in range(len(valid_data))]

for i, valid in enumerate(valid_data):
    if i % 10000 == 0:
        print("valid", i, len(valid_data))
    t = int(valid[-1])
    if latest_t != t:
        for ee in range(num_e):
            if len(entity_s_his_cache[ee]) != 0:
                if len(entity_s_his[ee]) >= history_len:
                    entity_s_his[ee].pop(0)
                    att_s_his[ee].pop(0)
                    rel_s_his[ee].pop(0)
                    self_att_s_his[ee].pop(0)

                entity_s_his[ee].append(entity_s_his_cache[ee].copy())
                att_s_his[ee].append(att_s_his_cache[ee].copy())
                rel_s_his[ee].append(rel_s_his_cache[ee].copy())
                self_att_s_his[ee].append(self_att_s_his_cache[ee].copy())

                entity_s_his_cache[ee] = []
                att_s_his_cache[ee] = []
                rel_s_his_cache[ee] = []
                self_att_s_his_cache[ee] = []

            if len(entity_o_his_cache[ee]) != 0:
                if len(entity_o_his[ee]) >= history_len:
                    entity_o_his[ee].pop(0)
                    att_o_his[ee].pop(0)
                    rel_o_his[ee].pop(0)
                    self_att_o_his[ee].pop(0)

                entity_o_his[ee].append(entity_o_his_cache[ee].copy())
                att_o_his[ee].append(att_o_his_cache[ee].copy())
                rel_o_his[ee].append(rel_o_his_cache[ee].copy())
                self_att_o_his[ee].append(self_att_o_his_cache[ee].copy())

                entity_o_his_cache[ee] = []
                att_o_his_cache[ee] = []
                rel_o_his_cache[ee] = []
                self_att_o_his_cache[ee] = []
        latest_t = t

    s = int(valid[0])
    r = int(valid[1])
    o = int(valid[2])
    att_s = [valid[3], valid[5]]
    att_o = [valid[4], valid[6]]

    entity_s_history_data_valid[i] = entity_s_his[s].copy()
    att_s_history_data_valid[i] = att_s_his[s].copy()
    rel_s_history_data_valid[i] = rel_s_his[s].copy()
    self_att_s_history_data_valid[i] = self_att_s_his[s].copy()

    entity_o_history_data_valid[i] = entity_o_his[o].copy()
    att_o_history_data_valid[i] = att_o_his[o].copy()
    rel_o_history_data_valid[i] = rel_o_his[o].copy()
    self_att_o_history_data_valid[i] = self_att_o_his[o].copy()

    if len(entity_s_his_cache[s]) == 0:
        entity_s_his_cache[s] = [o]
        rel_s_his_cache[s] = [r]
        att_s_his_cache[s] = [att_o]
        self_att_s_his_cache[s] = [att_s]
    else:
        entity_s_his_cache[s] = entity_s_his_cache[s] + [o]
        rel_s_his_cache[s] = rel_s_his_cache[s] + [r]
        att_s_his_cache[s] = att_s_his_cache[s] + [att_o]
        self_att_s_his_cache[s] = self_att_s_his_cache[s] + [att_s]

    if len(entity_o_his_cache[o]) == 0:
        entity_o_his_cache[o] = [s]
        rel_o_his_cache[o] = [r]
        att_o_his_cache[o] = [att_s]
        self_att_o_his_cache[o] = [att_o]
    else:
        entity_o_his_cache[o] = entity_o_his_cache[o] + [s]
        rel_o_his_cache[o] = rel_o_his_cache[o] + [r]
        att_o_his_cache[o] = att_o_his_cache[o] + [att_s]
        self_att_o_his_cache[o] = self_att_o_his_cache[o] + [att_o]
    # print(o_his_cache[o])

# entity_s_history_data_valid = [[list(c) for c in k]
#                                for k in entity_s_history_data_valid]
# rel_s_history_data_valid = [[list(c) for c in k]
#                             for k in rel_s_history_data_valid]
# att_s_history_data_valid = [[list(c) for c in k]
#                             for k in att_s_history_data_valid]
self_att_s_history_data_valid = [[c[0] for c in k]
                                 for k in self_att_s_history_data_valid]

# entity_o_history_data_valid = [[list(c) for c in k]
#                                for k in entity_o_history_data_valid]
# rel_o_history_data_valid = [[list(c) for c in k]
#                             for k in rel_o_history_data_valid]
# att_o_history_data_valid = [[list(c) for c in k]
#                             for k in att_o_history_data_valid]
self_att_o_history_data_valid = [[c[0] for c in k]
                                 for k in self_att_o_history_data_valid]

with open('valid_entity_s_history_data.json', 'w') as fp:
    json.dump(entity_s_history_data_valid, fp, indent=4)
with open('valid_rel_s_history_data.json', 'w') as fp:
    json.dump(rel_s_history_data_valid, fp, indent=4)
with open('valid_att_s_history_data.json', 'w') as fp:
    json.dump(att_s_history_data_valid, fp, indent=4)
with open('valid_self_att_s_history_data.json', 'w') as fp:
    json.dump(self_att_s_history_data_valid, fp, indent=4)

with open('valid_entity_o_history_data.json', 'w') as fp:
    json.dump(entity_o_history_data_valid, fp, indent=4)
with open('valid_rel_o_history_data.json', 'w') as fp:
    json.dump(rel_o_history_data_valid, fp, indent=4)
with open('valid_att_o_history_data.json', 'w') as fp:
    json.dump(att_o_history_data_valid, fp, indent=4)
with open('valid_self_att_o_history_data.json', 'w') as fp:
    json.dump(self_att_o_history_data_valid, fp, indent=4)

entity_s_history_data_test = [[] for _ in range(len(test_data))]
att_s_history_data_test = [[] for _ in range(len(test_data))]
rel_s_history_data_test = [[] for _ in range(len(test_data))]
self_att_s_history_data_test = [[] for _ in range(len(test_data))]

entity_o_history_data_test = [[] for _ in range(len(test_data))]
att_o_history_data_test = [[] for _ in range(len(test_data))]
rel_o_history_data_test = [[] for _ in range(len(test_data))]
self_att_o_history_data_test = [[] for _ in range(len(test_data))]

for i, test in enumerate(test_data):
    if i % 10000 == 0:
        print("test", i, len(test_data))
    t = int(test[-1])
    if latest_t != t:
        for ee in range(num_e):
            if len(entity_s_his_cache[ee]) != 0:
                if len(entity_s_his[ee]) >= history_len:
                    entity_s_his[ee].pop(0)
                    att_s_his[ee].pop(0)
                    rel_s_his[ee].pop(0)
                    self_att_s_his[ee].pop(0)

                entity_s_his[ee].append(entity_s_his_cache[ee].copy())
                att_s_his[ee].append(att_s_his_cache[ee].copy())
                rel_s_his[ee].append(rel_s_his_cache[ee].copy())
                self_att_s_his[ee].append(self_att_s_his_cache[ee].copy())

                entity_s_his_cache[ee] = []
                att_s_his_cache[ee] = []
                rel_s_his_cache[ee] = []
                self_att_s_his_cache[ee] = []

            if len(entity_o_his_cache[ee]) != 0:
                if len(entity_o_his[ee]) >= history_len:
                    entity_o_his[ee].pop(0)
                    att_o_his[ee].pop(0)
                    rel_o_his[ee].pop(0)
                    self_att_o_his[ee].pop(0)

                entity_o_his[ee].append(entity_o_his_cache[ee].copy())
                att_o_his[ee].append(att_o_his_cache[ee].copy())
                rel_o_his[ee].append(rel_o_his_cache[ee].copy())
                self_att_o_his[ee].append(self_att_o_his_cache[ee].copy())

                entity_o_his_cache[ee] = []
                att_o_his_cache[ee] = []
                rel_o_his_cache[ee] = []
                self_att_o_his_cache[ee] = []
        latest_t = t

    s = int(test[0])
    r = int(test[1])
    o = int(test[2])
    att_s = [test[3], test[5]]
    att_o = [test[4], test[6]]

    entity_s_history_data_test[i] = entity_s_his[s].copy()
    att_s_history_data_test[i] = att_s_his[s].copy()
    rel_s_history_data_test[i] = rel_s_his[s].copy()
    self_att_s_history_data_test[i] = self_att_s_his[s].copy()

    entity_o_history_data_test[i] = entity_o_his[o].copy()
    att_o_history_data_test[i] = att_o_his[o].copy()
    rel_o_history_data_test[i] = rel_o_his[o].copy()
    self_att_o_history_data_test[i] = self_att_o_his[o].copy()

    if len(entity_s_his_cache[s]) == 0:
        entity_s_his_cache[s] = [o]
        rel_s_his_cache[s] = [r]
        att_s_his_cache[s] = [att_o]
        self_att_s_his_cache[s] = [att_s]
    else:
        entity_s_his_cache[s] = entity_s_his_cache[s] + [o]
        rel_s_his_cache[s] = rel_s_his_cache[s] + [r]
        att_s_his_cache[s] = att_s_his_cache[s] + [att_o]
        self_att_s_his_cache[s] = self_att_s_his_cache[s] + [att_s]

    if len(entity_o_his_cache[o]) == 0:
        entity_o_his_cache[o] = [s]
        rel_o_his_cache[o] = [r]
        att_o_his_cache[o] = [att_s]
        self_att_o_his_cache[o] = [att_o]
    else:
        entity_o_his_cache[o] = entity_o_his_cache[o] + [s]
        rel_o_his_cache[o] = rel_o_his_cache[o] + [r]
        att_o_his_cache[o] = att_o_his_cache[o] + [att_s]
        self_att_o_his_cache[o] = self_att_o_his_cache[o] + [att_o]
    # print(o_his_cache[o])

# entity_s_history_data_test = [[list(c) for c in k]
#                               for k in entity_s_history_data_test]
# rel_s_history_data_test = [[list(c) for c in k]
#                            for k in rel_s_history_data_test]
# att_s_history_data_test = [[list(c) for c in k]
#                            for k in att_s_history_data_test]
self_att_s_history_data_test = [[c[0] for c in k]
                                for k in self_att_s_history_data_test]

# entity_o_history_data_test = [[list(c) for c in k]
#                               for k in entity_o_history_data_test]
# rel_o_history_data_test = [[list(c) for c in k]
#                            for k in rel_o_history_data_test]
# att_o_history_data_test = [[list(c) for c in k]
#                            for k in att_o_history_data_test]
self_att_o_history_data_test = [[c[0] for c in k]
                                for k in self_att_o_history_data_test]

with open('test_entity_s_history_data.json', 'w') as fp:
    json.dump(entity_s_history_data_test, fp, indent=4)
with open('test_rel_s_history_data.json', 'w') as fp:
    json.dump(rel_s_history_data_test, fp, indent=4)
with open('test_att_s_history_data.json', 'w') as fp:
    json.dump(att_s_history_data_test, fp, indent=4)
with open('test_self_att_s_history_data.json', 'w') as fp:
    json.dump(self_att_s_history_data_test, fp, indent=4)

with open('test_entity_o_history_data.json', 'w') as fp:
    json.dump(entity_o_history_data_test, fp, indent=4)
with open('test_rel_o_history_data.json', 'w') as fp:
    json.dump(rel_o_history_data_test, fp, indent=4)
with open('test_att_o_history_data.json', 'w') as fp:
    json.dump(att_o_history_data_test, fp, indent=4)
with open('test_self_att_o_history_data.json', 'w') as fp:
    json.dump(self_att_o_history_data_test, fp, indent=4)