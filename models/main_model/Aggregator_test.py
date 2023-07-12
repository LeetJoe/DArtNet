import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from utils_test import *


class MeanAggregator(nn.Module):
    def __init__(self, h_dim, dropout, seq_len=10):
        super(MeanAggregator, self).__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len

    def forward(self, s_hist, rel_hist, att_s_hist, self_att_s_hist, s, r,
                ent_embeds, rel_embeds, W1, W3, W4):

        '''
        print('forward params are here')
        print(s_hist)    # 2
        print(rel_hist)    # 3
        print(att_s_hist)    # 4
        print(self_att_s_hist)    # 5
        print(s)
        print(r)
        print(ent_embeds)
        print(rel_embeds)
        print(W1)
        print(W3)
        print(W4)
        print("\n\n")
        '''

        # for i in range(len(s_hist)):
        #     assert (len(s_hist[i]) == len(self_att_s_hist[i]))
        # print('forward')
        s_len_non_zero, \
            s_tem, \
            r_tem, \
            embeds_stack, \
            embeds_static_stack, \
            len_s, \
            s_idx = get_sorted_s_r_embed(  # debug here
            s_hist,    # 2
            rel_hist,    # 3
            att_s_hist,    # 4
            # self_att_s_hist,    # 5, 没有此项
            s,    # 1
            r,    # 1
            ent_embeds,
            rel_embeds,
            W1,
            W3,
            W4)
        # print('forward1')
        # embeds_stack 里的信息是 [att, s, rel]，embeds_static_stack 里的信息是 [s, rel]。这两个变量都是用 s_len 相同的方式做了排序的，所以后面
        # 才可以在转换之后使用 s_len_non_zero 进行 split。

        # s_len(注意不是len_s,len_s是未排序的长度值序列) 按元素项长度降序的长度值序列； s_len_non_zero, 前者去掉后面的0；s_tem 去掉空值，按长度降序排的 s； r_tem ... r；
        # embeds_stack, embeddings of att + s + r；embeds_static_stack, embedings of s + r；s_idx，s_len 中各项原来的位置；

        # To get mean vector at each time
        curr = 0
        rows = []   # rows 是 len_s 的展开，如果 len_s 在 i 位置的值是 5，就往 rows 里 append 5 个 i.
        cols = []   # cols 就是与 rows 长度一致的自然数序列,从0到len(rows)-1.
        for i, leng in enumerate(len_s):
            rows.extend([i] * leng)
            cols.extend(list(range(curr, curr + leng)))
            curr += leng
        rows = torch.LongTensor(rows)
        cols = torch.LongTensor(cols)
        # stack 把 rows 和 cols 叠在一起增加一个维度, 如 rows.shape=[4, 4], cols.shape=[4, 4], 则 idxes.shape=[2, 4, 4]
        idxes = torch.stack([rows, cols], dim=0)

        # mask_tensor.shape = [2556, 4178] 其中 idxes.shape=[2, 4178]和torch.ones(len(rows)).shape=[4178]，
        # 因为 rows 与 cols 分别表示 sparse coo 的行与列，因此 mask_tensor 的shape里的 2556 实际上是 rows.max()+1, 而 4178 则是cols.max()+1。
        # mask_tensor[i](i < 2556)逻辑上是一个长度为4178的向量，print(mask_tensor[i])显示为idxes和values两个list，两个list的长度是相同的。
        # 其含义是：idxes中的每一个元素j，表示mask_tensor[i]行里的一个下标j，而values里与idxes对应位置的元素k，即mask_tensor[i][j]=k是一个非0元素；
        # 故 j < len(mask_tensor[i]),且不在idxes里的其它 j 表示 mask_tensor[i][j]=0. 用这种方式来压缩表示 sparse matrix.
        # 这种表示方法是 sparse 压缩存储中的 coo 模式，即 coordinate matrix 模式。一般定义里，行数 m 与列数 n 是相等的，等于 nnz，但是这里应该是另一种存储方式，它的行数
        # m=n<=nnz=原矩阵的行数，如果一行中有多个非0元素，则对应的列作为 indices 是一个 list 而非数字，相对应的 values 也是一个 list。
        mask_tensor = torch.sparse.FloatTensor(idxes, torch.ones(len(rows)))
        mask_tensor = mask_tensor.cuda()

        # torch.sparse.mm 表示矩阵相乘, mask_tensor.shape=[2556, 4178], embeds_stack.shape=[4178,200], 结果是[2556,200]
        # 这里用到了 embeds_stack，里面包含 [att, s, rel] 信息，经 mask_tensor 把相关内容变换进了 embeds_sum 里。
        embeds_sum = torch.sparse.mm(mask_tensor, embeds_stack)
        # tensor a/b 表示 a 中每一行向量，除以 b 中每一行里的"单值向量"(b必须是二维的不能是一维的)里的值，即 b 的行数必须与 a 相同，但每一行只能有1个元素。
        # 这里的 mean 是按子元素里的项数进行平均，len_s s_hist 每一个元素的长度序列。
        embeds_mean = embeds_sum / torch.Tensor(len_s).cuda().view(-1, 1)
        # embed_means 的行数正好等于 s_len_non_zero.sum()，这里用 split 把 embeds_mean 按 s_len_non_zero 里的数字进行拆分。
        embeds_split = torch.split(embeds_mean, s_len_non_zero.tolist())

        # 上面使用 mask_tensor 处理 embeds_stack, 这里使用同样的处理过程处理 embeds_static_stack。
        embeds_static_sum = torch.sparse.mm(mask_tensor, embeds_static_stack)
        embeds_static_mean = embeds_static_sum / torch.Tensor(
            len_s).cuda().view(-1, 1)
        embeds_static_split = torch.split(embeds_static_mean,
                                          s_len_non_zero.tolist())
        # 现在得到的 embeds_split 与 embeds_static_split 是 tuple 类型的数据，其中的元素是 tensor 类型，而 tensor.shape=[len, 200]，其中
        # len 即是前面 split 时的分切长度，也即是 s_len_non_zero 里的对应值。tuple 的长度与 s_len_non_zero 的长度相同，这里是 450.

        # 下面这两个都是 shape=[450,10,600], 注意这里 self.seq_len=10 是通过 --seq_len 传进来的，它与 data 目录里的 gen_history.py 里的
        # history_len 是一致的，它约束了一条路径的最大(回溯)长度。下面这两个全初始化为 0 的数组实际是两个容器，第二个维度对应的就是这个路径的长度。
        s_embed_seq_tensor = torch.zeros(len(s_len_non_zero), self.seq_len,
                                         3 * self.h_dim).cuda()
        att_embed_seq_tensor = torch.zeros(len(s_len_non_zero), self.seq_len,
                                           3 * self.h_dim).cuda()

        for i in range(len(embeds_split)):

            # 取 embeds_split 与 embeds_static_split 中的第 i 个 tensor 元素，shape=[len, 200](上面分析过)。
            embeds = embeds_split[i]
            embeds_static = embeds_static_split[i]

            # s_tem 和 r_tem 是 s 和 r 按 idx 重排得到的结果，idx 与 s_len_non_zero 是一个顺序， s_tem[i] 返回的是一个 node id。
            # ent_embeds 和 rel_embeds 是 DArtNet 初始化时生成的 torch.nn.parameter.Parameter Layer。
            # ent_embeds.shape=[num_nodes, 200], rel_embeds.shape=[num_rels, 200], num_nodes 与 num_rels 取自 stat.txt， 200 取自超参 n_hidden。
            # ent_embeds[s_tem[i]] 是 node s_tem[i] 的 hidden vector, 一个长度为 200 的向量。按第二维度(第二个参数1) repeat 之后得到的是 shape=[len, 200], 即以列为准在行上复制了 len 次。
            self_e_embed = ent_embeds[s_tem[i]].repeat(len(embeds), 1)
            self_r_embed = rel_embeds[r_tem[i]].repeat(len(embeds), 1)

            # self_att_s_hist 没有像 s_len_non_zero 那样重新排序，所以要找到对应位置的数据需要使用 s_idx[i]. 其中 self_att_s_hist[s_idx[i]]) 是一个
            # 一维向量，经 view(-1, 1) 后变成一个 [att_len, 1] 的矩阵，再经 W1() 之后变成一个 [att_len, 200] 的矩阵，再经 relu(rectified linear unit)
            # 处理，得到的 self_att_embed 是一个 [att_len, 200] 的矩阵。
            self_att_embed = F.relu(
                W1(torch.tensor(self_att_s_hist[s_idx[i]]).to(torch.float32).cuda().view(-1, 1)))

            # torch.cat([a, b], dim=1) 表示把列表里的矩阵按维度 1 堆叠，如 a.shape=[1,2,3], b.shape=[1,1,3]，结果是 [1,3,6]
            # 在此循环里，对每一个 i，将 cat() 的结果 [len, 600] 放进 [i, 10, 600] 里，显然 len<=10.
            att_embed_seq_tensor[i, torch.arange(len(embeds)), :] = torch.cat(
                [self_att_embed, self_e_embed, embeds], dim=1)

            s_embed_seq_tensor[i, torch.arange(len(embeds)), :] = torch.cat(
                [self_e_embed, self_r_embed, embeds_static], dim=1)

        # 对自己进行 dropout。
        s_embed_seq_tensor = self.dropout(s_embed_seq_tensor)
        att_embed_seq_tensor = self.dropout(att_embed_seq_tensor)

        # torch.nn.utils.rnn.pack_padded_sequence(input, length, batch_first=False, enforce_sorted=True), 这个可以说是矩阵组织的核心目标了，
        # input 是一个三维数组，一般形式为 shape=[T, B, *] 其中 B 表示的是 batch size；而 T 则表示一个 batch 里的有效数据 shape=[t, *]
        # 里 t 的最大值，这样 [T, B, *] 才能容纳下这一批次数据的 batch_size 个 [t, *]. batch_first 如果为 True，则表示 input.shape = [B, T, *]
        # 即 B 与 T 的位置互换，正如在这里的情况。enforce_sorted 默认为 True 要求 [B, T, *] 里的 [t, *] 按 t 由大到小排序，这就是为什么前面会把 s 和 rel 等
        # 全部按照 len 从大到小重排的原因。至于 length 这个参数，实际上就是 s_len_non_zero，显然 T = s_len_non_zero[0]。
        # 而最后得到的是 s_packed_input 和 s_packed_input 的 shape 都是 [2556, 600]，因此 pack_padded_sequence 的实际作用应该就是：
        # 按照 length 里给出的长度序列，把 [T, B, *] 里的全 0 的数据去掉得到 [SUM(t0,t1,...tB-1), *], 显然有 SUM() = length.sum()，这里有 SUM()=2556, 且 *=600.
        s_packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            s_embed_seq_tensor, s_len_non_zero.to('cpu'), batch_first=True)
        att_packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            att_embed_seq_tensor, s_len_non_zero.to('cpu'), batch_first=True)

        return s_packed_input, att_packed_input

    def predict(self, s_history, rel_history, att_history, self_att_history, s,
                r, ent_embeds, rel_embeds, W1, W3, W4):
        inp_s = torch.zeros(len(s_history), 3 * self.h_dim).cuda()
        inp_att = torch.zeros(len(s_history), 3 * self.h_dim).cuda()
        for i, s_o in enumerate(s_history):
            r_o = rel_history[i]
            a_o = att_history[i]
            if type(a_o) != torch.Tensor:
                a_o = torch.tensor(a_o, dtype=torch.float)

            self_a_o = self_att_history[i]
            if type(self_a_o) != torch.Tensor:
                self_a_o = torch.tensor(self_a_o, dtype=torch.float)

            e_s = ent_embeds[s_o].view(-1, self.h_dim)
            e_r = rel_embeds[r_o].view(-1, self.h_dim)
            e_att = F.relu(W1(a_o.type(torch.FloatTensor).cuda().view(
                -1, 1))).view(-1, self.h_dim)
            # e_s_att = F.relu(W2(torch.cat([e_att, e_s], dim=-1)))
            e = F.relu(W3(torch.cat([e_att, e_s, e_r], dim=-1)))
            e_static = F.relu(W4(torch.cat([e_s, e_r], dim=-1)))
            tem = torch.mean(e, dim=0)
            tem_static = torch.mean(e_static, dim=0)
            # print(self_a_o)
            e_self_att = F.relu(W1(self_a_o.cuda().view(1,
                                                        1))).view(self.h_dim)

            inp_s[i] = torch.cat([
                ent_embeds[s].view(self.h_dim), rel_embeds[r].view(self.h_dim),
                tem_static
            ],
                                 dim=0)
            inp_att[i] = torch.cat(
                [e_self_att, ent_embeds[s].view(self.h_dim), tem], dim=0)
        return inp_s, inp_att
