import numpy as np
import os
from collections import defaultdict
import pickle
import torch
from pprint import pprint
import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 返回一个6元组
def load_hexaruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        hexapleList = []
        times = set()

        # 数据从左到右依次是：头，关系，尾，头属性，尾属性，时间
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
    times.sort()

    return np.asarray(hexapleList), np.asarray(times)

# 第一个返回是实体（边？）数量，第二个是关系数量
def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

# xxx_data里也包含了时间
train_data, train_times = load_hexaruples('', 'train.txt')
test_data, test_times = load_hexaruples('', 'test.txt')
valid_data, valid_times = load_hexaruples('', 'valid.txt')
# total_data, _ = load_hexaruples('', 'train.txt', 'test.txt')

history_len = 10
# 实体数与关系数
num_e, num_r = get_total_number('', 'stat.txt')

# todo s和下面的o代表什么？s似乎是指head, o是指tail

# 长度等于实体总数
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

# 长度等于训练集
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

# 长度等于实体总数
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
    t = int(train[5])
    if latest_t != t:   # 时间不匹配则进入，然后 *_cache 就会被清空

        # 长度等于num_e的那些空数组，num_e循环量级
        for ee in range(num_e):

            # for head
            if len(entity_s_his_cache[ee]) != 0:
                if len(entity_s_his[ee]) >= history_len:
                    # 过长则从头pop
                    # todo 每次append的数量可能大于1，但是这里每次只pop 1个，有可能 len(entity_s_his[ee])会一直 > history_len
                    entity_s_his[ee].pop(0)
                    att_s_his[ee].pop(0)
                    rel_s_his[ee].pop(0)
                    self_att_s_his[ee].pop(0)

                # 把cache区的数据append到后面
                entity_s_his[ee].append(entity_s_his_cache[ee].copy())
                att_s_his[ee].append(att_s_his_cache[ee].copy())
                rel_s_his[ee].append(rel_s_his_cache[ee].copy())
                self_att_s_his[ee].append(self_att_s_his_cache[ee].copy())

                # cache区清空
                entity_s_his_cache[ee] = []
                att_s_his_cache[ee] = []
                rel_s_his_cache[ee] = []
                self_att_s_his_cache[ee] = []

            # for tail
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

    # [s, r, o, att_s, att_o, t] 是第 i 个 train 的元
    # *_data[i] 构成所有与 i 具有相同的 head 的六元组的相关信息按时间排序的一个序列
    # entity_s_history_data 里放的是 i 之前的同"头"六元组的 tail 序列；
    entity_s_history_data[i] = entity_s_his[s].copy()
    # att_s_history_data 里放的是 i 之前的同"头"六元组的 tail_attr 序列；
    att_s_history_data[i] = att_s_his[s].copy()
    # rel_s_history_data 里放的是 i 之前的同"头"六元组的 relation 序列；
    rel_s_history_data[i] = rel_s_his[s].copy()
    # self_att_s_history_data 里放的是 i 之前的同"头"六元组的 head_attr 序列；
    self_att_s_history_data[i] = self_att_s_his[s].copy()


    # entity_o_history_data 里放的是 i 之前的同"尾"六元组的 head 序列；
    entity_o_history_data[i] = entity_o_his[o].copy()
    # att_o_history_data 里放的是 i 之前的同"尾"六元组的 head_attr 序列；
    att_o_history_data[i] = att_o_his[o].copy()
    # rel_o_history_data 里放的是 i 之前的同"尾"六元组的 relation 序列；
    rel_o_history_data[i] = rel_o_his[o].copy()
    # self_att_o_history_data 里放的是 i 之前的同"尾"六元组的 tail_attr 序列；
    self_att_o_history_data[i] = self_att_o_his[o].copy()
    # print(o_history_data_g[i])

    # todo test 可以直接把 *_cache 初始化为 np.array([], dtype='int')，这样就不需要检查长度为0了，直接concatenate就行了。
    # 当latest_t == t，即连续的两个train item的time相同的时候，*_cache 不会被清空，可能会出现*_cache里的列表>1的情况。
    # 只有当时间发生变化的时候，latest_t 才更新，即数据是按时间做了排序的，相同的时间只会连续出现。
    if len(entity_s_his_cache[s]) == 0:
        # 所以*_cache的下标是head，对应的元素是tail构成的数组
        entity_s_his_cache[s] = np.array([o])
        rel_s_his_cache[s] = np.array([r])
        att_s_his_cache[s] = np.array([att_o])
        self_att_s_his_cache[s] = np.array([att_s])
    else:
        # 加入
        entity_s_his_cache[s] = np.concatenate((entity_s_his_cache[s], [o]),
                                               axis=0)
        rel_s_his_cache[s] = np.concatenate((rel_s_his_cache[s], [r]), axis=0)
        att_s_his_cache[s] = np.concatenate((att_s_his_cache[s], [att_o]),
                                            axis=0)
        self_att_s_his_cache[s] = np.concatenate(
            (self_att_s_his_cache[s], [att_s]), axis=0)

    # todo 跟前面一样
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

    # todo 最后一组 *_cache 没有加入到 *_his 中，*_data 里也没有这部分数据。

    # print(s_history_data[i], s_history_data_g[i])
    # with open('ttt.txt', 'wb') as fp:
    #     pickle.dump(s_history_data_g, fp)
    # print("save")

# with open('train_graphs.txt', 'wb') as fp:
#     pickle.dump(graph_dict_train, fp)

# entity_s_history_data = [[list(c) for c in k] for k in entity_s_history_data]
# rel_s_history_data = [[list(c) for c in k] for k in rel_s_history_data]
# att_s_history_data = [[list(c) for c in k] for k in att_s_history_data]

# self_att_s_history_data 所有子项（对应位置的 att_o 序列）的第一个，构成的一个 list；
self_att_s_history_data = [[c[0] for c in k] for k in self_att_s_history_data]

# entity_o_history_data = [[list(c) for c in k] for k in entity_o_history_data]
# rel_o_history_data = [[list(c) for c in k] for k in rel_o_history_data]
# att_o_history_data = [[list(c) for c in k] for k in att_o_history_data]

# self_att_o_history_data 所有子项（对应位置的 att_s 序列）的第一个，构成的一个 list；
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