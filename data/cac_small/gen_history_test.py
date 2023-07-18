import numpy as np
import os
from collections import defaultdict
import pickle
import dgl
import torch
from pprint import pprint

# 返回一个由结构为[head, relation, tail, attribute_head, attribute_tail, time]的项构成的 list 和一个由 time 构成的 list。
def load_hexaruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        hexapleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            att_head = float(line_split[3])
            att_tail = float(line_split[4])
            time = int(line_split[5])
            hexapleList.append([head, rel, tail, att_head, att_tail, time])
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
                att_head = float(line_split[3])
                att_tail = float(line_split[4])
                time = int(line_split[5])
                hexapleList.append([head, rel, tail, att_head, att_tail, time])
                times.add(time)
    times = list(times)
    times.sort() # todo 不明白这里为什么要 sort() 一下，虽然这个返回实际上是没用的，至少在这个脚本里是没用的。

    return np.asarray(hexapleList), np.asarray(times)


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)


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

# xxx_s/o_his 这些数据以 node 值为下标组织为一种映射关系，所以其长度为 num_e
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
e = []
r = []

latest_t = 0

# xxx_s/o_his_cache 这些数据以 node 值为下标组织为一种映射关系，所以其长度为 num_e
entity_s_his_cache = [[] for _ in range(num_e)]
att_s_his_cache = [[] for _ in range(num_e)]
rel_s_his_cache = [[] for _ in range(num_e)]
self_att_s_his_cache = [[] for _ in range(num_e)]

entity_o_his_cache = [[] for _ in range(num_e)]
att_o_his_cache = [[] for _ in range(num_e)]
rel_o_his_cache = [[] for _ in range(num_e)]
self_att_o_his_cache = [[] for _ in range(num_e)]

# xxx_s/o_history_data 这些数据以 train_data 里六元组的 index 为下标组织，所以长度是 len(train_data)
entity_s_history_data = [[] for _ in range(len(train_data))]
att_s_history_data = [[] for _ in range(len(train_data))]
rel_s_history_data = [[] for _ in range(len(train_data))]
self_att_s_history_data = [[] for _ in range(len(train_data))]

entity_o_history_data = [[] for _ in range(len(train_data))]
att_o_history_data = [[] for _ in range(len(train_data))]
rel_o_history_data = [[] for _ in range(len(train_data))]
self_att_o_history_data = [[] for _ in range(len(train_data))]


for i, train in enumerate(train_data):
    if i % 10000 == 0:
        print("train", i, len(train_data))
    # if i == 10000:
    #     break
    t = int(train[5])
    if latest_t != t:
        # 每次 time 发生变化的时候，对 xxx_s/o_his_cache 中的数据进行处理：
        # 1. 如果 xxx_s/o_his_cache[ee] 里有数据（通常是一个list，里面多数时候只有一个值，也有些会有多个值），将其 append 到 xxx_s/o_his[ee] 后面；
        # 2. 在 append 之前检查 xxx_s/o_his[ee] 的长度，如果 >= history_len，那就把最早加入其中的数据 pop 一个，也就是 xxx_s/o_his[ee] 的长度最大不超过 history_len，去旧留新；
        # 3. 最后把 xxx_s/o_his_cache[ee] 的内容置空，准备接收下一个 time 段里的内容，直到 time 再次变化时，重复执行上面的操作。
        # 这样 xxx_s/o_his[ee] 里存放的是 ee 相关的一个序列，序列中的每一项（是一个列表）代表在某个时间里的相关路径，这个序列按时间升序排列。

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
    att_s = train[3]
    att_o = train[4]
    # print(s_his[r][s])

    # train_data 是按时间排序的，所以第 i 项时 xxx_s/o_his[s] 里的数据都是 i 之前的；
    # 当某个 j > i 时有 sj==si 时，xxx_s/o_his[s] 里的数据可能：变得更长(如果没超过 len_history); 左移(如果超过了 len_history,旧的数据左边移出，新的数据右侧进入)
    # xxx_s/o_history_data[i] 里记录了到 i 位置时，xxx_s/o_his[s] 里的快照。
    # 由于只有发生时间变化时 xxx_s/o_his 的内容才会更新，因此在 t 不变时的 i 到 k 里若有 si==sk，则有 xxx_s/o_history_data[i]==xxx_s/o_history_data[k]
    entity_s_history_data[i] = entity_s_his[s].copy()
    att_s_history_data[i] = att_s_his[s].copy()
    rel_s_history_data[i] = rel_s_his[s].copy()
    self_att_s_history_data[i] = self_att_s_his[s].copy()

    entity_o_history_data[i] = entity_o_his[o].copy()
    att_o_history_data[i] = att_o_his[o].copy()
    rel_o_history_data[i] = rel_o_his[o].copy()
    self_att_o_history_data[i] = self_att_o_his[o].copy()
    # print(o_history_data_g[i])

    # 在 t 不变的6元组序列里，数据按 s/o 的值为下标组织在 xxx_s/o_his_cache[s/o] 里，有多个的话则形成一个 list，这个 list 最后在 t
    # 发生改变的时候会作为一个 item append 到 xxx_s/o_hist[s/o] 里。
    if len(entity_s_his_cache[s]) == 0:
        entity_s_his_cache[s] = np.array([o])
        rel_s_his_cache[s] = np.array([r])
        att_s_his_cache[s] = np.array([att_o])
        self_att_s_his_cache[s] = np.array([att_s])
    else:
        entity_s_his_cache[s] = np.concatenate((entity_s_his_cache[s], [o]),
                                               axis=0)
        rel_s_his_cache[s] = np.concatenate((rel_s_his_cache[s], [r]), axis=0)
        att_s_his_cache[s] = np.concatenate((att_s_his_cache[s], [att_o]),
                                            axis=0)
        self_att_s_his_cache[s] = np.concatenate(
            (self_att_s_his_cache[s], [att_s]), axis=0)

    if len(entity_o_his_cache[o]) == 0:
        entity_o_his_cache[o] = np.array([s])
        rel_o_his_cache[o] = np.array([r])
        att_o_his_cache[o] = np.array([att_s])
        self_att_o_his_cache[o] = np.array([att_o])
    else:
        entity_o_his_cache[o] = np.concatenate((entity_o_his_cache[o], [s]),
                                               axis=0)
        rel_o_his_cache[o] = np.concatenate((rel_o_his_cache[o], [r]), axis=0)
        att_o_his_cache[o] = np.concatenate((att_o_his_cache[o], [att_s]),
                                            axis=0)
        self_att_o_his_cache[o] = np.concatenate(
            (self_att_o_his_cache[o], [att_o]), axis=0)

    # print(s_history_data[i], s_history_data_g[i])
    # with open('ttt.txt', 'wb') as fp:
    #     pickle.dump(s_history_data_g, fp)
    # print("save")

# with open('train_graphs.txt', 'wb') as fp:
#     pickle.dump(graph_dict_train, fp)

# k 是组织在 xxx_s/o_history_data 里的第 i(s,r,o 6元组) 个与 xxx 相对应的 xxx_s/o_his[s/o] 在遍历到i时的快照。
# c 是在 xxx_s/o_his[s/o] 里的项，即通过 xxx_s/o_his_cache[s/o] 组织而来的 list，c[0] 则是 list 里的第一项。
# 因为 t 是相同的，只能说第 1 项对应的6元组的 i 排在其它项的前面，因为 t 粒度的问题，不能确定 list 内部是否也是按时间排序的。

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
# pprint(att_s_history_data)
# pprint(self_att_s_history_data)

with open('train_entity_s_history_data.txt', 'wb') as fp:
    pickle.dump(entity_s_history_data, fp)
with open('train_rel_s_history_data.txt', 'wb') as fp:
    pickle.dump(rel_s_history_data, fp)
with open('train_att_s_history_data.txt', 'wb') as fp:
    pickle.dump(att_s_history_data, fp)
with open('train_self_att_s_history_data.txt', 'wb') as fp:
    pickle.dump(self_att_s_history_data, fp)

with open('train_entity_o_history_data.txt', 'wb') as fp:
    pickle.dump(entity_o_history_data, fp)
with open('train_rel_o_history_data.txt', 'wb') as fp:
    pickle.dump(rel_o_history_data, fp)
with open('train_att_o_history_data.txt', 'wb') as fp:
    pickle.dump(att_o_history_data, fp)
with open('train_self_att_o_history_data.txt', 'wb') as fp:
    pickle.dump(self_att_o_history_data, fp)

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
    t = int(valid[5])
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
    att_s = valid[3]
    att_o = valid[4]

    entity_s_history_data_valid[i] = entity_s_his[s].copy()
    att_s_history_data_valid[i] = att_s_his[s].copy()
    rel_s_history_data_valid[i] = rel_s_his[s].copy()
    self_att_s_history_data_valid[i] = self_att_s_his[s].copy()

    entity_o_history_data_valid[i] = entity_o_his[o].copy()
    att_o_history_data_valid[i] = att_o_his[o].copy()
    rel_o_history_data_valid[i] = rel_o_his[o].copy()
    self_att_o_history_data_valid[i] = self_att_o_his[o].copy()

    if len(entity_s_his_cache[s]) == 0:
        entity_s_his_cache[s] = np.array([o])
        rel_s_his_cache[s] = np.array([r])
        att_s_his_cache[s] = np.array([att_o])
        self_att_s_his_cache[s] = np.array([att_s])
    else:
        entity_s_his_cache[s] = np.concatenate((entity_s_his_cache[s], [o]),
                                               axis=0)
        rel_s_his_cache[s] = np.concatenate((rel_s_his_cache[s], [r]), axis=0)
        att_s_his_cache[s] = np.concatenate((att_s_his_cache[s], [att_o]),
                                            axis=0)
        self_att_s_his_cache[s] = np.concatenate(
            (self_att_s_his_cache[s], [att_s]), axis=0)

    if len(entity_o_his_cache[o]) == 0:
        entity_o_his_cache[o] = np.array([s])
        rel_o_his_cache[o] = np.array([r])
        att_o_his_cache[o] = np.array([att_s])
        self_att_o_his_cache[o] = np.array([att_o])
    else:
        entity_o_his_cache[o] = np.concatenate((entity_o_his_cache[o], [s]),
                                               axis=0)
        rel_o_his_cache[o] = np.concatenate((rel_o_his_cache[o], [r]), axis=0)
        att_o_his_cache[o] = np.concatenate((att_o_his_cache[o], [att_s]),
                                            axis=0)
        self_att_o_his_cache[o] = np.concatenate(
            (self_att_o_his_cache[o], [att_o]), axis=0)
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

with open('valid_entity_s_history_data.txt', 'wb') as fp:
    pickle.dump(entity_s_history_data_valid, fp)
with open('valid_rel_s_history_data.txt', 'wb') as fp:
    pickle.dump(rel_s_history_data_valid, fp)
with open('valid_att_s_history_data.txt', 'wb') as fp:
    pickle.dump(att_s_history_data_valid, fp)
with open('valid_self_att_s_history_data.txt', 'wb') as fp:
    pickle.dump(self_att_s_history_data_valid, fp)

with open('valid_entity_o_history_data.txt', 'wb') as fp:
    pickle.dump(entity_o_history_data_valid, fp)
with open('valid_rel_o_history_data.txt', 'wb') as fp:
    pickle.dump(rel_o_history_data_valid, fp)
with open('valid_att_o_history_data.txt', 'wb') as fp:
    pickle.dump(att_o_history_data_valid, fp)
with open('valid_self_att_o_history_data.txt', 'wb') as fp:
    pickle.dump(self_att_o_history_data_valid, fp)

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
    t = int(test[5])
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
    att_s = test[3]
    att_o = test[4]

    entity_s_history_data_test[i] = entity_s_his[s].copy()
    att_s_history_data_test[i] = att_s_his[s].copy()
    rel_s_history_data_test[i] = rel_s_his[s].copy()
    self_att_s_history_data_test[i] = self_att_s_his[s].copy()

    entity_o_history_data_test[i] = entity_o_his[o].copy()
    att_o_history_data_test[i] = att_o_his[o].copy()
    rel_o_history_data_test[i] = rel_o_his[o].copy()
    self_att_o_history_data_test[i] = self_att_o_his[o].copy()

    if len(entity_s_his_cache[s]) == 0:
        entity_s_his_cache[s] = np.array([o])
        rel_s_his_cache[s] = np.array([r])
        att_s_his_cache[s] = np.array([att_o])
        self_att_s_his_cache[s] = np.array([att_s])
    else:
        entity_s_his_cache[s] = np.concatenate((entity_s_his_cache[s], [o]),
                                               axis=0)
        rel_s_his_cache[s] = np.concatenate((rel_s_his_cache[s], [r]), axis=0)
        att_s_his_cache[s] = np.concatenate((att_s_his_cache[s], [att_o]),
                                            axis=0)
        self_att_s_his_cache[s] = np.concatenate(
            (self_att_s_his_cache[s], [att_s]), axis=0)

    if len(entity_o_his_cache[o]) == 0:
        entity_o_his_cache[o] = np.array([s])
        rel_o_his_cache[o] = np.array([r])
        att_o_his_cache[o] = np.array([att_s])
        self_att_o_his_cache[o] = np.array([att_o])
    else:
        entity_o_his_cache[o] = np.concatenate((entity_o_his_cache[o], [s]),
                                               axis=0)
        rel_o_his_cache[o] = np.concatenate((rel_o_his_cache[o], [r]), axis=0)
        att_o_his_cache[o] = np.concatenate((att_o_his_cache[o], [att_s]),
                                            axis=0)
        self_att_o_his_cache[o] = np.concatenate(
            (self_att_o_his_cache[o], [att_o]), axis=0)
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

with open('test_entity_s_history_data.txt', 'wb') as fp:
    pickle.dump(entity_s_history_data_test, fp)
with open('test_rel_s_history_data.txt', 'wb') as fp:
    pickle.dump(rel_s_history_data_test, fp)
with open('test_att_s_history_data.txt', 'wb') as fp:
    pickle.dump(att_s_history_data_test, fp)
with open('test_self_att_s_history_data.txt', 'wb') as fp:
    pickle.dump(self_att_s_history_data_test, fp)

with open('test_entity_o_history_data.txt', 'wb') as fp:
    pickle.dump(entity_o_history_data_test, fp)
with open('test_rel_o_history_data.txt', 'wb') as fp:
    pickle.dump(rel_o_history_data_test, fp)
with open('test_att_o_history_data.txt', 'wb') as fp:
    pickle.dump(att_o_history_data_test, fp)
with open('test_self_att_o_history_data.txt', 'wb') as fp:
    pickle.dump(self_att_o_history_data_test, fp)