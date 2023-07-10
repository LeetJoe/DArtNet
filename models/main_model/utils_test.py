import numpy as np
import os
import torch
import torch.nn.functional as F


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def load_hexaruples(inPath, fileName, fileName2=None, fileName3=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            att_head = float(line_split[3])
            att_tail = float(line_split[4])
            time = int(line_split[5])
            quadrupleList.append([head, rel, tail, att_head, att_tail, time])
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
                quadrupleList.append(
                    [head, rel, tail, att_head, att_tail, time])
                times.add(time)
    if fileName3 is not None:
        with open(os.path.join(inPath, fileName3), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                att_head = float(line_split[3])
                att_tail = float(line_split[4])
                time = int(line_split[5])
                quadrupleList.append(
                    [head, rel, tail, att_head, att_tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)


def load_quadruples(inPath, fileName, fileName2=None, fileName3=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[5])
            quadrupleList.append([head, rel, tail, time])
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
                time = int(line_split[5])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    if fileName3 is not None:
        with open(os.path.join(inPath, fileName3), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[5])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)


def make_batch(a, b, c, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(a), n):
        # Create an index range for l of n items:
        yield a[i:i + n], b[i:i + n], c[i:i + n]


def make_batch2(a, b, c, d, e, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(a), n):
        # Create an index range for l of n items:
        yield a[i:i + n], b[i:i + n], c[i:i + n], d[i:i + n], e[i:i + n]


def make_batch3(a, b, c, d, e, f, g, h, w, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(a), n):
        # Create an index range for l of n items:
        yield a[i:i + n], \
            b[i:i + n], \
            c[i:i + n], \
            d[i:i + n], \
            e[i:i + n], \
            f[i:i + n], \
            g[i:i + n], \
            h[i:i + n], \
            w[i:i + n]


def get_data(s_hist, o_hist):
    data = None
    for i, s_his in enumerate(s_hist):
        if len(s_his) != 0:
            # print(s_his)
            tem = torch.cat((torch.LongTensor([i]).repeat(
                len(s_his), 1), torch.LongTensor(s_his.cpu())),
                            dim=1)
            if data is None:
                data = tem.cpu().numpy()
            else:
                data = np.concatenate((data, tem.cpu().numpy()), axis=0)

    for i, o_his in enumerate(o_hist):
        if len(o_his) != 0:
            tem = torch.cat((torch.LongTensor(o_his[:, 1].cpu()).view(
                -1, 1), torch.LongTensor(o_his[:, 0].cpu()).view(
                    -1, 1), torch.LongTensor([i]).repeat(len(o_his), 1)),
                            dim=1)
            if data is None:
                data = tem.cpu().numpy()
            else:
                data = np.concatenate((data, tem.cpu().numpy()), axis=0)
    data = np.unique(data, axis=0)
    return data


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor


'''
Get sorted s and r to make batch for RNN (sorted by length)
'''


def get_sorted_s_r_embed(s_hist,     # 2
                         rel_hist,     # 3
                         att_hist,     # 4
                         s,
                         r,
                         ent_embeds,
                         rel_embeds,
                         W1,
                         W3,
                         W4):
    s_hist_len = torch.LongTensor(list(map(len, s_hist))).cuda()
    s_len, s_idx = s_hist_len.sort(0, descending=True)
    # torch.nonzero(s_len) 按顺序返回 s_len 中非 0 无素的下标
    num_non_zero = len(torch.nonzero(s_len))
    # 因为 s_len 已经按元素个数降序排列，如果其中有个数为 0 的子元素，那它一定排在后面，如下使用 num_non_zero 即可以得到 s_len 中非空子元素个数列表
    s_len_non_zero = s_len[:num_non_zero]

    s_hist_sorted = []
    rel_hist_sorted = []
    att_hist_sorted = []
    # 按排序结果，将三都重排放在新的变量组里。
    # 这里令 s_idx = s_idx[:num_non_zero]，那后面不就不需要再截断了？
    for idx in s_idx:
        s_hist_sorted.append(s_hist[idx.item()])
        rel_hist_sorted.append(rel_hist[idx.item()])
        att_hist_sorted.append(att_hist[idx.item()])

    flat_s = []
    flat_rel = []
    flat_att = []
    len_s = []

    # 在这里截断？直接对 s_idx 截断然后再按剩下的 s_idx 重组不就行了，为什么要分两步？
    s_hist_sorted = s_hist_sorted[:num_non_zero]
    rel_hist_sorted = rel_hist_sorted[:num_non_zero]
    att_hist_sorted = att_hist_sorted[:num_non_zero]

    for i in range(len(s_hist_sorted)):
        for j in range(len(s_hist_sorted[i])):
            # 长度序列
            len_s.append(len(s_hist_sorted[i][j]))
            for k in range(len(s_hist_sorted[i][j])):
                # 打平了的 s, rel, attr 序列
                flat_s.append(s_hist_sorted[i][j][k])
                flat_rel.append(rel_hist_sorted[i][j][k])
                flat_att.append(att_hist_sorted[i][j][k])

    # 以与 s_idx 的元素相对应的 s 的下标作排序，比如 s 是[1, 2, 3], s_idx 是[2,1,0], 那么 s[s_idx] = [3, 2, 1]
    s_tem = s[s_idx]
    r_tem = r[s_idx]

    # 对 id 进行 embedding。看来 embedding 不止是可以对自然语言的字词进行，任何可以视为特征的东西都可以使用 embedding？
    embeds_s = ent_embeds[torch.LongTensor(flat_s).cuda()]
    embeds_rel = rel_embeds[torch.LongTensor(flat_rel).cuda()]

    # view(-1, 1) 表示重新组织一个数组，比如 flat_att=[1,2,3,4,5,6,7,8], torch.tensor(flat_att).view(-1, 1) 会将其转化成一个
    # 8 * 1 的二维数据，等效于 view(8, 1); 而 view(4, 2) 则会将其转化为 4*2 的数组。如果 view 的两个参数相乘不等于 len 会报错。
    test_input = torch.tensor(flat_att).view(-1, 1).cuda()
    test_input = test_input.to(torch.float32)

    # W1 is Linear(in_features=1, out_features=200, bias=True)
    embeds_att = F.relu(
        W1(     # todo error here
            test_input
        )
    )

    embeds = F.relu(W3(torch.cat([embeds_att, embeds_s, embeds_rel], dim=1).to(torch.float32)))
    # embeds_split = torch.split(embeds, len_s)

    embeds_static = F.relu(W4(torch.cat([embeds_s, embeds_rel], dim=1).to(torch.float32)))
    # embeds_static_split = torch.split(embeds_static, len_s)

    return s_len_non_zero, s_tem, r_tem, embeds, embeds_static, len_s, s_idx  # embeds_split, embeds_static_split,
