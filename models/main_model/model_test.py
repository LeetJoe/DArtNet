import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from Aggregator_test import MeanAggregator
from utils import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class DArtNet(nn.Module):
    def __init__(self,
                 num_nodes,    # 1
                 h_dim,    # 2
                 num_rels,    # 3
                 dropout=0,    # 4
                 model=0,    # 5
                 seq_len=10,    # 6
                 num_k=10,    # 7
                 gamma=1):    # 8
        super(DArtNet, self).__init__()

        # 总 node(entity) 数量
        self.num_nodes = num_nodes    # 1

        # hidden stat dim, 对应 --n-hidden 参数
        self.h_dim = h_dim    # 2

        # 总 edge 数量
        self.num_rels = num_rels    # 3

        # dropout todo 跟优化有关,是一个 hyper parameter
        self.dropout = nn.Dropout(dropout)    # 4

        # todo 好像没什么用？
        self.model = model    # 5

        # 对应 args.seq_len, 默认 10
        self.seq_len = seq_len    # 6

        # 对应 args.num_k，cuttoff position, 对结果中取最符合分类的前 k 个数据。
        # torch.topk()：来求tensor中某个dim的前k大或者前k小的值以及对应的index
        self.num_k = num_k    # 7

        # loss 的系数，用于：loss = loss_sub + self.gamma * loss_att_sub
        self.gamma = gamma    # 8

        # 初始化全 0 的 entity embeddings, shape = num_nodes(stat.txt里的第一个数) * h_dim
        self.ent_embeds = nn.Parameter(torch.Tensor(self.num_nodes, self.h_dim))
        # 对 rel_embeds 进行"Glorot initialization"。
        nn.init.xavier_uniform_(self.ent_embeds,
                                gain=nn.init.calculate_gain('relu'))

        # 初始化全 0 的 relation embeddings, shape = num_rels(stat.txt里的第二个数) * h_dim
        self.rel_embeds = nn.Parameter(torch.Tensor(self.num_rels, self.h_dim))
        # 对 rel_embeds 进行"Glorot initialization"。
        nn.init.xavier_uniform_(self.rel_embeds,
                                gain=nn.init.calculate_gain('relu'))

        # GRU: multi-layer gated recurrent unit (GRU) RNN
        # 3 * self.h_dim 是 input size, self.h_dim 是 hidden size
        # 继承关系： GRU -> RNNBase -> Module
        # GRU 生成的矩阵内容不为 0，生成结果与 seed 有关。seed 不变，初始化的 sub_encoder 的内容是一样的。
        # 这里的 batch_first 与 torch.nn.utils.rnn.pack_padded_sequence 里的 batch_first 含义是一样的。
        self.sub_encoder = nn.GRU(3 * self.h_dim, self.h_dim, batch_first=True)
        # self.ob_encoder = self.sub_encoder

        # 同上，随机生的 att_encoder 与 seed 有关，但与 sub_encoder 不同。
        self.att_encoder = nn.GRU(3 * self.h_dim, self.h_dim, batch_first=True)

        # callable, 对应于 MeanAggregator 的 forward 方法（继承自 nn.Module 接口）
        self.aggregator_s = MeanAggregator(self.h_dim, dropout, seq_len)
        # self.aggregator_o = self.aggregator_s

        # nn.Linear()也是一个 callable 对象，调用的是 forward 方法（继承自 nn.Module )
        # 2 * self.h_dim 是 in_features, 1 是 out_features。
        # 以与 seed 有关的方式初始化为0的tensor，与 GRU 类似
        self.f1 = nn.Linear(2 * self.h_dim, 1)

        self.f2 = nn.Linear(3 * self.h_dim, self.num_nodes)

        # todo 在相关文章里经常提到的 W 矩阵？
        self.W1 = nn.Linear(1, self.h_dim)
        # self.W1 is Linear(in_features=1, out_features=200, bias=True)
        # self.W2 = nn.Linear(2 * self.h_dim, self.h_dim)
        self.W3 = nn.Linear(3 * self.h_dim, self.h_dim)
        self.W4 = nn.Linear(2 * self.h_dim, self.h_dim)

        # For recording history in inference

        self.entity_s_his_test = None
        self.att_s_his_test = None
        self.rel_s_his_test = None
        self.self_att_s_his_test = None

        # self.entity_o_his_test = None
        # self.att_o_his_test = None
        # self.rel_o_his_test = None
        # self.self_att_o_his_test = None

        self.entity_s_his_cache = None
        self.att_s_his_cache = None
        self.rel_s_his_cache = None
        self.self_att_s_his_cache = None

        # self.entity_o_his_cache = None
        # self.att_o_his_cache = None
        # self.rel_o_his_cache = None
        # self.self_att_o_his_cache = None

        self.att_s_dict = {}
        # self.att_o_dict = {}

        self.latest_time = 0

        # 两种损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.att_criterion = nn.MSELoss()  # mean squared error
        # 似乎凡是 nn.XXX() 这样的操作(nn.Parameters()除外)都相当于自动将构建的 layer 加入到 model 中. 因此执行 model.train()，会依次对
        # Dropout, GRU*2, MeanAggregator(Dropout), Linear*5, CrossEntropyLoss, MSELoss 进行 train(). 可通过 print(model) 查看。

    """
    Prediction function in training. 
    This should be different from testing because in testing we don't use ground-truth history.
    """
    def forward(self,
                triplets,    # 1, 即 batch data
                s_hist,    # 2
                rel_s_hist,    # 3
                att_s_hist,    # 4
                self_att_s_hist,    # 5
                o_hist,    # 6
                rel_o_hist,    # 7
                att_o_hist,    # 8
                self_att_o_hist,    # 9
                predict_both=True):
        # print('here')
        s = triplets[:, 0].type(torch.cuda.LongTensor)
        r = triplets[:, 1].type(torch.cuda.LongTensor)
        o = triplets[:, 2].type(torch.cuda.LongTensor)
        a_s = triplets[:, 3].type(torch.cuda.FloatTensor)
        a_o = triplets[:, 4].type(torch.cuda.FloatTensor)

        batch_size = len(s)

        # s_hist 的子元素的长度形成的列表
        s_hist_len = torch.LongTensor(list(map(len, s_hist))).cuda()

        # s_len 是对 s_hist_len 进行降序排序， s_idx 是排序后新的顺序里对应各元素在 s_hist_len 原下标
        s_len, s_idx = s_hist_len.sort(0, descending=True)

        # o_hist_len = torch.LongTensor(list(map(len, o_hist))).cuda()
        # o_len, o_idx = o_hist_len.sort(0, descending=True)
        # print('here1')

        # 这里的 s_packed_input 是由 [s, rel] 得来的，att_s_packed_input 是由 [att, s, rel] 得到的。
        # 其中 s_packed_input.data 可以得到其中的 tensor 数据，这里是 shape=[2556, 600]. 另外 s_packed_input.batch_sizes.sum() = 2556. att_s_packed_input 也是这样。
        # todo: 这个 batch_sizes 不知道是什么含义，因为其 sum() 与 s_len_non_zero.sum() 刚好相等，但是没找到两者之间的关联规律。
        s_packed_input, att_s_packed_input = self.aggregator_s(    # callable, type MeanAggregator()
            s_hist,  # 2
            rel_s_hist,  # 3
            att_s_hist,  # 4
            self_att_s_hist,  # 5
            s,  # 1 第 0 列，类型由 dtype=torch.float64 转换成了 torch.cuda.LongTensor
            r,  # 1 第 1 列，类型由 dtype=torch.float64 转换成了 torch.cuda.LongTensor
            self.ent_embeds,
            self.rel_embeds,
            self.W1,
            self.W3,
            self.W4)

        # o_packed_input, att_o_packed_input = self.aggregator_o(
        #     o_hist, rel_o_hist, att_o_hist, self_att_o_hist, o, r,
        #     self.ent_embeds, self.rel_embeds[self.num_rels:], self.W1, self.W2,
        #     self.W3, self.W4)

        # 是否执行预测
        if predict_both:
            # self.sub_encoder 是一个 GRU layer, 这里进行的是一个 RNN 操作。
            # 得到的 _(实际上就是output) 是一个与 s_packed_input 同类型的，但是它的 data.shape=[2556, 200], 第二个维度变了；batch_sizes是相同的。
            # 这里 s_packed_input(就是input) 是 2556 * 600, self.sub_encoder 是 600 * 200, 按矩阵乘应该得到 2556 * 200，这里得到的是 450 * 200.
            # input size = 600, output size = 200. 这里的 s_h 其实就是 h_n，h_0为0，GRU 的工作方式里，h_0 传入下一次运算得到 h_1,...直到最后得到 h_n。
            # h_i.shape=[450, 200], 故 s_h.shape=[450, 200], _ 才是真正的 output, todo 不知道为什么这里不用 ouput 而只用 h_n？
            _, s_h = self.sub_encoder(s_packed_input) # GRU 的输入实际应该由两部分组成，input 和 h_0，h_0 表示上一层传来的 h_i，这里是第一步故省略。
            # torch.squeeze() 把 shape 中的一维维度删除，如 shape=[4,1,5,1,6]，squeeze() 之将变成 [4,5,6]
            s_h = s_h.squeeze()
            # 这一步把是 s_h 补齐为 [500(=batch size), 200]，比如现在 s_h.shape=[450, 200], 要在维度 0 上叠加一个 shape=[50, 200] 的0矩阵。
            s_h = torch.cat(
                (s_h, torch.zeros(len(s) - len(s_h), self.h_dim).cuda()),
                dim=0)
            # 这里得到的 ob_pred.shape=[500, 90]，实际上就是这 500 个数据的预测，90 是解空间，ob_pred[i]的 0~89 下标中的数据表示预测结果是相应位置的 node 的概率。
            # self.f2 是 [600, 90] 的变换。
            ob_pred = self.f2(
                self.dropout(
                    # s[s_idx] 是按 s_idx 排序的 s，r[s_idx] 也是；分别使用 self.ent_embeds 和 self.rel_embeds 转化为 shape=[500, 200]
                    # 的 embeddings ，再和补齐为 shape=[500, 200] 的 s_h 进行 torch.cat()，得到 shape=[500, 600]. dropout() 操作不改变 shape，
                    # 经 self.f2(nn.Linear, 600 * 90) 层后得到 ob_pred.shape=[500, 90]
                    torch.cat((self.ent_embeds[s[s_idx]], s_h,
                               self.rel_embeds[r[s_idx]]),
                              dim=1)))

            # o[s_idx] 是按 s_idx 排序的 o，实际上就是 500 个 tail 的值；
            # ob_pred.shape=[500, 90], 每一项是一个长度为 90(与 node 总个数相同) 的向量，表示预测结果是相应 node id 的概率。
            # 两者使用交叉熵损失来计算 loss。
            loss_sub = self.criterion(ob_pred, o[s_idx])
        else:
            ob_pred = None
            loss_sub = 0

        # _, o_h = self.ob_encoder(o_packed_input)
        # (同上面的 s_h 的生成过程)self.att_encoder 是一个 GRU layer, 这里进行的是一个 RNN 操作。
        # 得到的 _ 是一个与 att_s_packed_input 同类型的，但是它的 data.shape=[2556, 200], 第二个维度变了；batch_sizes是相同的。
        # att_s_h.shape=[450, 200]
        _, att_s_h = self.att_encoder(att_s_packed_input)
        # _, att_o_h = self.att_encoder(att_o_packed_input)
        # print('here2')

        # o_h = o_h.squeeze()
        att_s_h = att_s_h.squeeze()
        # att_o_h = att_o_h.squeeze()

        # o_h = torch.cat(
        #     (o_h, torch.zeros(len(o) - len(o_h), self.h_dim).cuda()), dim=0)
        att_s_h = torch.cat(
            (att_s_h, torch.zeros(len(s) - len(att_s_h), self.h_dim).cuda()),
            dim=0)
        # att_o_h = torch.cat(
        #     (att_o_h, torch.zeros(len(o) - len(att_o_h), self.h_dim).cuda()),
        #     dim=0)
        # print('here3')

        # 与 predict 那里不同，这里只使用了 self.ent_embeds[s[s_idx]], 没有使用 self.rel_embeds[r[s_idx]]。而且 predict_both 那里也没有使用 squeeze()。
        # self.f1 是 [400, 1] 的变换，最终得到的 sub_att_pred.shape=[500]，其中每一个值代表对 self_att 的预测.
        sub_att_pred = self.f1(
            self.dropout(torch.cat((self.ent_embeds[s[s_idx]], att_s_h),
                                   dim=1))).squeeze()

        # sub_pred = self.f2(
        #     self.dropout(
        #         torch.cat((self.ent_embeds[o[o_idx]], o_h,
        #                    self.rel_embeds[self.num_rels:][r[o_idx]]),
        #                   dim=1)))

        # ob_att_pred = self.f1(
        #     self.dropout(torch.cat((self.ent_embeds[o[o_idx]], att_o_h),
        #                            dim=1))).squeeze()

        # loss_ob = self.criterion(sub_pred, s[o_idx])

        # att_criterion, mean square error, MSELoss
        # 注意这里用的是 MSELoss，不是交叉熵损失。sub_att_pred 与 a_s[s_idx] 的 shape 都是 [500]
        loss_att_sub = self.att_criterion(sub_att_pred, a_s[s_idx])
        # loss_att_ob = self.att_criterion(ob_att_pred, a_o[o_idx])

        # final loss，传参时 self.gamma 示例给的值是 1，也就是两者同地位相加得到最终的 loss
        loss = loss_sub + self.gamma * loss_att_sub

        return loss, loss_att_sub, ob_pred, sub_att_pred, s_idx

    def init_history(self):
        self.entity_s_his_test = [[] for _ in range(self.num_nodes)]
        self.att_s_his_test = [[] for _ in range(self.num_nodes)]
        self.rel_s_his_test = [[] for _ in range(self.num_nodes)]
        self.self_att_s_his_test = [[] for _ in range(self.num_nodes)]

        # self.entity_o_his_test = [[] for _ in range(self.num_nodes)]
        # self.att_o_his_test = [[] for _ in range(self.num_nodes)]
        # self.rel_o_his_test = [[] for _ in range(self.num_nodes)]
        # self.self_att_o_his_test = [[] for _ in range(self.num_nodes)]

        self.entity_s_his_cache = [[] for _ in range(self.num_nodes)]
        self.att_s_his_cache = [[] for _ in range(self.num_nodes)]
        self.rel_s_his_cache = [[] for _ in range(self.num_nodes)]
        self.self_att_s_his_cache = [[] for _ in range(self.num_nodes)]

        # self.entity_o_his_cache = [[] for _ in range(self.num_nodes)]
        # self.att_o_his_cache = [[] for _ in range(self.num_nodes)]
        # self.rel_o_his_cache = [[] for _ in range(self.num_nodes)]
        # self.self_att_o_his_cache = [[] for _ in range(self.num_nodes)]

    def get_loss(self,
                 triplets,    # 1
                 s_hist,    # 2
                 rel_s_hist,    # 3
                 att_s_hist,    # 4
                 self_att_s_hist,    # 5
                 o_hist,    # 6
                 rel_o_hist,    # 7
                 att_o_hist,    # 8
                 self_att_o_hist):    # 9
        print("before model.forward....")
        # forward 的本质是计算 loss, 后三个 _,_,_ 实际上是  ob_pred, sub_att_pred, s_idx，这后面没用到。
        loss, loss_att_sub, _, _, _ = self.forward(triplets,    # 1
                                                   s_hist,    # 2
                                                   rel_s_hist,     # 3
                                                   att_s_hist,    # 4
                                                   self_att_s_hist,     # 5
                                                   o_hist,    # 6
                                                   rel_o_hist,     # 7
                                                   att_o_hist,    # 8
                                                   self_att_o_hist)    # 9
        print("after forward....")
        return loss, loss_att_sub

    """
    Prediction function in testing
    """
    def predict(self, triplets, s_hist, rel_s_hist, att_s_hist,
                self_att_s_hist, o_hist, rel_o_hist, att_o_hist,
                self_att_o_hist):

        self.att_s_dict = {}
        # self.att_o_dict = {}
        self.att_residual_dict = {}

        # 得到预测xx_pred和损失loss_xxx，只用 att_* 部分，loss 和 ob_pred 这里都用不到。
        # 注意这里最后的参数 False, 与 get_loss() 不同，这里 predict_both=False,即 ob_pred 不执行
        _, loss_att_sub, _, sub_att_pred, s_idx = self.forward(
            triplets, s_hist, rel_s_hist, att_s_hist, self_att_s_hist, o_hist,
            rel_o_hist, att_o_hist, self_att_o_hist, False)
        # print(triplets[:, 0])
        # print(s_hist)
        # print(sub_att_pred)
        indices = {}
        for i in range(len(triplets)):
            s = triplets[s_idx[i], 0].type(torch.LongTensor).item()
            o = triplets[s_idx[i], 2].type(torch.LongTensor).item()
            t = triplets[s_idx[i], 5].type(torch.LongTensor).item()
            s_att = sub_att_pred[i].cpu().item()

            if s not in self.att_s_dict: # 如果是第一次预测，保存预测 att 值
                self.att_s_dict[s] = s_att  # s_att 是预测值
                indices[s] = i  # 记录 s 的下标
            else: # 如果不是第一次预测，要求新的预测结果要与旧的预测结果一致
                if (self.att_s_dict[s] != s_att):
                    print(s, " ", self.att_s_dict[s], " ", s_att)
                # 这里的断言是要求在这一批次的数据中，s 原先有过的预测值需要与后来再次出现的预测值相同。
                # todo 多次预测的值可能相近但不可能完全相同吧，这里为什么要做这种断言？
                assert (abs(self.att_s_dict[s]-s_att) < 0.0000001)

            # s = triplets[o_idx[i], 0].type(torch.LongTensor).item()
            # o = triplets[o_idx[i], 2].type(torch.LongTensor).item()
            # t = triplets[o_idx[i], 5].type(torch.LongTensor).item()
            # o_att = ob_att_pred[i].cpu().item()

            # if o not in self.att_o_dict:
            #     self.att_o_dict[o] = o_att
            # else:
            #     assert (self.att_o_dict[o] == o_att)

        # 每次 predict() 执行的时候，都把这个 self.att_residual_dict 置空，然后在本轮预测结束之后，在 self.att_s_dict 里尚未被预测过的
        # 节点会使用一个空的 s_h 进行预测然后放进 self.att_residual_dict 里面，todo 似乎是用来在更新 cache 的时候占位使用。
        for i in range(self.num_nodes):
            if i not in self.att_s_dict:  # and i not in self.att_o_dict:
                s_h = torch.zeros(1, self.h_dim).cuda()
                sub_att_pred = self.f1(
                    torch.cat((self.ent_embeds[[i]], s_h), dim=1)).squeeze()
                self.att_residual_dict[i] = sub_att_pred

        return loss_att_sub

    # 单三元组预测,这里实际上只预测了 head 那一半，tail 那一半没有预测
    def predict_single(self, triplet, s_hist, rel_s_hist, att_s_hist,
                       self_att_s_hist, o_hist, rel_o_hist, att_o_hist,
                       self_att_o_hist):
        # print(triplet)
        s = triplet[0].type(torch.cuda.LongTensor)
        r = triplet[1].type(torch.cuda.LongTensor)
        o = triplet[2].type(torch.cuda.LongTensor)
        a_s = triplet[3].type(torch.cuda.FloatTensor)
        a_o = triplet[4].type(torch.cuda.FloatTensor)
        t = triplet[5].cpu()
        # print('here')
        # 这里对 xxx_s_his_cache 的构建不是在此方法里完成，而且通过对此方法的多次调用由历史数据构建而成。
        if self.latest_time != t:

            for ee in range(self.num_nodes):
                if len(self.entity_s_his_cache[ee]) != 0:
                    if len(self.entity_s_his_test[ee]) >= self.seq_len:
                        self.entity_s_his_test[ee].pop(0)
                        self.att_s_his_test[ee].pop(0)
                        self.self_att_s_his_test[ee].pop(0)
                        self.rel_s_his_test[ee].pop(0)

                    self.entity_s_his_test[ee].append(
                        self.entity_s_his_cache[ee].clone())
                    self.att_s_his_test[ee].append(
                        self.att_s_his_cache[ee].clone())
                    self.self_att_s_his_test[ee].append(
                        self.self_att_s_his_cache[ee])
                    self.rel_s_his_test[ee].append(
                        self.rel_s_his_cache[ee].clone())

                    self.entity_s_his_cache[ee] = []
                    self.att_s_his_cache[ee] = []
                    self.self_att_s_his_cache[ee] = []
                    self.rel_s_his_cache[ee] = []

                # if len(self.entity_o_his_cache[ee]) != 0:
                #     if len(self.entity_o_his_test[ee]) >= self.seq_len:
                #         self.entity_o_his_test[ee].pop(0)
                #         self.att_o_his_test[ee].pop(0)
                #         self.self_att_o_his_test[ee].pop(0)
                #         self.rel_o_his_test[ee].pop(0)

                #     self.entity_o_his_test[ee].append(
                #         self.entity_o_his_cache[ee].clone())
                #     self.att_o_his_test[ee].append(
                #         self.att_o_his_cache[ee].clone())
                #     self.self_att_o_his_test[ee].append(
                #         self.self_att_o_his_cache[ee])
                #     self.rel_o_his_test[ee].append(
                #         self.rel_o_his_cache[ee].clone())

                #     self.entity_o_his_cache[ee] = []
                #     self.att_o_his_cache[ee] = []
                #     self.self_att_o_his_cache[ee] = []
                #     self.rel_o_his_cache[ee] = []

            self.latest_time = t

        # 如果有历史数据，通过历史数据计算出 s_h；如果没有，置 s_h 为0.
        if len(s_hist) == 0:
            s_h = torch.zeros(self.h_dim).cuda()

        else:
            if len(self.entity_s_his_test[s]) == 0:
                self.entity_s_his_test[s] = s_hist.copy()
                self.rel_s_his_test[s] = rel_s_hist.copy()
                self.att_s_his_test[s] = att_s_hist.copy()
                self.self_att_s_his_test[s] = self_att_s_hist

            s_history = self.entity_s_his_test[s]
            rel_s_history = self.rel_s_his_test[s]
            att_s_history = self.att_s_his_test[s]
            self_att_s_history = self.self_att_s_his_test[s]

            # inp_s 与 _ 分别是以 i 为下标组织的 I_ht 和 A_ht., 用作 GRU 的输入。
            inp_s, _ = self.aggregator_s.predict(
                s_history, rel_s_history, att_s_history, self_att_s_history, s,
                r, self.ent_embeds, self.rel_embeds, self.W1, self.W3, self.W4)

            # GRU encoder，与 train 里的操作一样。
            _, s_h = self.sub_encoder(
                inp_s.view(1, len(s_history), 3 * self.h_dim))
            s_h = s_h.squeeze()

        # if len(o_hist) == 0:
        #     o_h = torch.zeros(self.h_dim).cuda()
        # else:
        #     if len(self.entity_o_his_test[o]) == 0:
        #         self.entity_o_his_test[o] = o_hist.copy()
        #         self.rel_o_his_test[o] = rel_o_hist.copy()
        #         self.att_o_his_test[o] = att_o_hist.copy()
        #         self.self_att_o_his_test[o] = self_att_o_hist

        #     o_history = self.entity_o_his_test[o]
        #     rel_o_history = self.rel_o_his_test[o]
        #     att_o_history = self.att_o_his_test[o]
        #     self_att_o_history = self.self_att_o_his_test[o]

        #     inp_o, _ = self.aggregator_o.predict(
        #         o_history, rel_o_history, att_o_history, self_att_o_history, o,
        #         r, self.ent_embeds, self.rel_embeds[self.num_rels:], self.W1,
        #         self.W2, self.W3, self.W4)

        #     _, o_h = self.ob_encoder(
        #         inp_o.view(1, len(o_history), 3 * self.h_dim))
        #     o_h = o_h.squeeze()

        # 与 forward() 里的过程类似，先得到 s_h，然后与 ent_embeds 和 rel_embeds 拼接再经 Linear 得到预测结果。
        # ob_pred.shape=[90], 每一个 item() 表示对应的 node 符合预测结果的概率。是对 tail 的预测
        ob_pred = self.f2(
            torch.cat((self.ent_embeds[s], s_h, self.rel_embeds[r]), dim=0))
        # sub_pred = self.f2(
        #     torch.cat(
        #         (self.ent_embeds[o], o_h, self.rel_embeds[self.num_rels:][r]),
        #         dim=0))


        # _ 中是 topk probability, o_candidate 是相应的下标，这里就是 node id.
        _, o_candidate = torch.topk(ob_pred, self.num_k)
        # _, s_candidate = torch.topk(sub_pred, self.num_k)

        # todo 更新 cache，这里为什么要更新？ cache 会更新一些数据进 test 系列数据，这样或许会使用预测结果更准确？
        self.entity_s_his_cache[s], self.rel_s_his_cache[
            s], self.att_s_his_cache[s], self.self_att_s_his_cache[
                s] = self.update_cache(self.entity_s_his_cache[s],
                                       self.rel_s_his_cache[s],
                                       self.att_s_his_cache[s],
                                       self.self_att_s_his_cache[s], s.cpu(),
                                       r.cpu(), o_candidate.cpu())
        # self.entity_o_his_cache[o], self.rel_o_his_cache[
        #     o], self.att_o_his_cache[o], self.self_att_o_his_cache[
        #         o] = self.update_cache(self.entity_o_his_cache[o],
        #                                self.rel_o_his_cache[o],
        #                                self.att_o_his_cache[o],
        #                                self.self_att_o_his_cache[o], o.cpu(),
        #                                r.cpu(), s_candidate.cpu())

        # loss_sub = self.criterion(ob_pred.view(1, -1), o.view(-1))
        # loss_ob = self.criterion(sub_pred.view(1, -1), s.view(-1))

        # loss = loss_sub + loss_ob

        return ob_pred

    # triplet 及后面的数据都是单条记录的（all_triplets 除外，它是所有 train + validate 数据）
    def evaluate_filter(self, triplet, s_hist, rel_s_hist, att_s_hist,
                        self_att_s_hist, o_hist, rel_o_hist, att_o_hist,
                        self_att_o_hist, all_triplets):
        s = triplet[0].type(torch.cuda.LongTensor)
        r = triplet[1].type(torch.cuda.LongTensor)
        o = triplet[2].type(torch.cuda.LongTensor)
        # print(s_hist)
        # print(rel_s_hist)

        # 得到的是对 tail 的预测，是一个 shape=[90] 的tensor
        ob_pred = self.predict_single(triplet, s_hist, rel_s_hist, att_s_hist,
                                      self_att_s_hist, o_hist, rel_o_hist,
                                      att_o_hist, self_att_o_hist)
        o_label = o
        s_label = s
        # sub_pred = torch.sigmoid(sub_pred), 经过 sigmoid 方法之后， ob_pred.shape 不变
        ob_pred = torch.sigmoid(ob_pred)

        # ob_pred 长度等于 node num，表示各个位置的节点作为此关系中 tail 的位置的概率，这里是要判断 o 的概率，它可能不是最大的，但是只要大于一个临界值便可采用。
        # 以真实的 o 的预测概率作为 "ground", 是一个"下限"/"保底"
        ground = ob_pred[o].clone()

        # torch.nonzero() 对 all_triplets[:, 0](即所有三元组中的s) 中的s进行检查，如果等于这里的 s，则将对应位置的下标作为行加入矩阵中，最后得到一个 shape=[x, 1]
        # 的矩阵，其中 x 是 s 出现的次数. 再使用 view(-1) 将其变成一个列表得到 s_id，即所有 validate+train 数据6元组中，head=s 的那些在 validate+train 集合中的位置。
        s_id = torch.nonzero(
            all_triplets[:, 0].type(torch.cuda.LongTensor) == s).to('cpu').view(-1)
        # 与前面类似，这里找的是 validate+train 所有6元组中，head=s, 且 rel=r 的那些位置，放在 idx 中。
        idx = torch.nonzero(
            all_triplets[s_id, 1].type(torch.cuda.LongTensor) == r).to('cpu').view(-1)
        # 对 s_id 按 idx 进行筛选。
        idx = s_id[idx]

        # 找到 validate+train 与 idx 对应位置的 o, 放进 idx 里，此时 idx 不再是下标，而是原 idx 所表示的下标处的 o
        idx = all_triplets[idx, 2].type(torch.cuda.LongTensor)
        ob_pred[idx] = 0     # 将这些位置置 0 了, idx 里包含 o。
        ob_pred[o_label] = ground      # 然后又单独把 o 的概率恢复了。
        # 前面这一段的作用是，把所有 validate+train 里的数据中，s 和 r 与 triplet 相等的那些数据里的 tail 数据都找出来，然后把 ob_pred
        # 里面对这些 tail 的预测概率置为0，然后单独把 triplet 里的 tail 的概率保留。ob_pred 的下标 i 的值表示的是预测 o 是 i 的概率。

        # ob_pred_comp1.shape=[90],找到 ob_pred 里概率大于 ground 的那些位置，也就是预测的 [s,r,o] 中，[s,r,o] 实际并不存在于数据集中，但是对 o 的预测概率却大于
        # triplet 中实际的 o 的那些 node id, 对 ob_pred_comp1，有 ob_pred_comp1[node_id]=True.
        ob_pred_comp1 = (ob_pred > ground).data.cpu().numpy()
        # ob_pred_comp2.shape=[90],找到 ob_pred 里概率等于 ground 的那些位置(node id), ob_pred_comp2[node_id]=True
        ob_pred_comp2 = (ob_pred == ground).data.cpu().numpy()

        # np.sum(ob_pred_comp1) 得到的是 ob_pred 里面 p>ground 的数量，np.sum(ob_pred_comp2) 是 p=ground，np.sum(ob_pred_comp2) - 1 是为了除去 o 本身，
        # todo 但是再除以 2 是为什么？然后这两部分相加再 +1 ??
        rank_ob = np.sum(ob_pred_comp1) + (
            (np.sum(ob_pred_comp2) - 1.0) / 2) + 1
        # rank_ob = 预测概率大于 ground 的 node id 的数量 + （预测概率等于 ground 的 node id 的数量 - 1）/ 2 + 1

        # ground = sub_pred[s].clone()

        # o_id = torch.nonzero(
        #     all_triplets[:, 2].type(torch.cuda.LongTensor) == o).view(-1)
        # idx = torch.nonzero(
        #     all_triplets[o_id, 1].type(torch.cuda.LongTensor) == r).view(-1)
        # idx = o_id[idx]
        # idx = all_triplets[idx, 0].type(torch.cuda.LongTensor)
        # sub_pred[idx] = 0
        # sub_pred[s_label] = ground

        # sub_pred_comp1 = (sub_pred > ground).data.cpu().numpy()
        # sub_pred_comp2 = (sub_pred == ground).data.cpu().numpy()
        # rank_sub = np.sum(sub_pred_comp1) + (
        #     (np.sum(sub_pred_comp2) - 1.0) / 2) + 1
        return np.array([rank_ob])

    def update_cache(self, s_his_cache, r_his_cache, att_his_cache,
                     self_att_his_cache, s, r, o_candidate):
        if len(s_his_cache) == 0:
            s_his_cache = o_candidate.view(-1)
            r_his_cache = r.repeat(len(o_candidate), 1).view(-1)
            att_his_cache = []
            for key in s_his_cache:
                k = key.item()
                if k in self.att_s_dict:
                    att_his_cache.append(self.att_s_dict[k])
                # elif k in self.att_o_dict:
                #     att_his_cache.append(self.att_o_dict[k])
                else:
                    att_his_cache.append(self.att_residual_dict[k])

            if s.item() in self.att_s_dict:
                self_att_his_cache = self.att_s_dict[s.item()]
            # elif s.item() in self.att_o_dict:
            #     self_att_his_cache = self.att_o_dict[s.item()]
            else:
                self_att_his_cache = self.att_residual_dict[s.item()]

            if type(att_his_cache) != torch.Tensor:
                att_his_cache = torch.FloatTensor(att_his_cache)
        else:
            ent_list = s_his_cache[torch.nonzero(r_his_cache == r).view(-1)]
            tem = []
            for i in range(len(o_candidate)):
                if o_candidate[i] not in ent_list:
                    tem.append(i)

            if len(tem) != 0:
                forward = o_candidate[torch.LongTensor(tem)].view(-1)
                forward2 = r.repeat(len(tem), 1).view(-1)

                s_his_cache = torch.cat(
                    (torch.LongTensor(s_his_cache), forward), dim=0)
                r_his_cache = torch.cat(
                    (torch.LongTensor(r_his_cache), forward2), dim=0)
                att_his_cache = torch.cat((torch.FloatTensor(att_his_cache),
                                           forward2.type(torch.FloatTensor)),
                                          dim=0)
                # self_att_his_cache = torch.cat((self_att_his_cache, forward2),
                #                                dim=0)
                # print('---------------no')
                for i in range(len(s_his_cache)):
                    if s_his_cache[i] in ent_list:
                        # print('-------------------yes')
                        if s_his_cache[i].item() in self.att_s_dict:
                            att_his_cache[i] = self.att_s_dict[
                                s_his_cache[i].item()]
                        # elif s_his_cache[i].item() in self.att_o_dict:
                        #     att_his_cache[i] = self.att_o_dict[
                        #         s_his_cache[i].item()]
                        else:
                            att_his_cache[i] = self.att_residual_dict[
                                s_his_cache[i].item()]

                if s.item() in self.att_s_dict:
                    self_att_his_cache = self.att_s_dict[s.item()]
                # elif s.item() in self.att_o_dict:
                #     self_att_his_cache = self.att_o_dict[s.item()]
                else:
                    self_att_his_cache = self.att_residual_dict[s.item()]

        return s_his_cache, r_his_cache, att_his_cache, self_att_his_cache
