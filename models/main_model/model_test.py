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

        if predict_both:
            _, s_h = self.sub_encoder(s_packed_input)
            s_h = s_h.squeeze()
            s_h = torch.cat(
                (s_h, torch.zeros(len(s) - len(s_h), self.h_dim).cuda()),
                dim=0)
            ob_pred = self.f2(
                self.dropout(
                    torch.cat((self.ent_embeds[s[s_idx]], s_h,
                               self.rel_embeds[r[s_idx]]),
                              dim=1)))
            loss_sub = self.criterion(ob_pred, o[s_idx])
        else:
            ob_pred = None
            loss_sub = 0

        # _, o_h = self.ob_encoder(o_packed_input)

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

        loss_att_sub = self.att_criterion(sub_att_pred, a_s[s_idx])
        # loss_att_ob = self.att_criterion(ob_att_pred, a_o[o_idx])

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
        # todo forward 的本质是计算 loss ？？
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

            if s not in self.att_s_dict:
                self.att_s_dict[s] = s_att
                indices[s] = i
            else:
                assert (self.att_s_dict[s] == s_att)

            # s = triplets[o_idx[i], 0].type(torch.LongTensor).item()
            # o = triplets[o_idx[i], 2].type(torch.LongTensor).item()
            # t = triplets[o_idx[i], 5].type(torch.LongTensor).item()
            # o_att = ob_att_pred[i].cpu().item()

            # if o not in self.att_o_dict:
            #     self.att_o_dict[o] = o_att
            # else:
            #     assert (self.att_o_dict[o] == o_att)

        for i in range(self.num_nodes):
            if i not in self.att_s_dict:  # and i not in self.att_o_dict:
                s_h = torch.zeros(1, self.h_dim).cuda()
                sub_att_pred = self.f1(
                    torch.cat((self.ent_embeds[[i]], s_h), dim=1)).squeeze()
                self.att_residual_dict[i] = sub_att_pred

        return loss_att_sub

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

            inp_s, _ = self.aggregator_s.predict(
                s_history, rel_s_history, att_s_history, self_att_s_history, s,
                r, self.ent_embeds, self.rel_embeds, self.W1, self.W3, self.W4)

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

        ob_pred = self.f2(
            torch.cat((self.ent_embeds[s], s_h, self.rel_embeds[r]), dim=0))
        # sub_pred = self.f2(
        #     torch.cat(
        #         (self.ent_embeds[o], o_h, self.rel_embeds[self.num_rels:][r]),
        #         dim=0))

        _, o_candidate = torch.topk(ob_pred, self.num_k)
        # _, s_candidate = torch.topk(sub_pred, self.num_k)

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

    def evaluate_filter(self, triplet, s_hist, rel_s_hist, att_s_hist,
                        self_att_s_hist, o_hist, rel_o_hist, att_o_hist,
                        self_att_o_hist, all_triplets):
        s = triplet[0].type(torch.cuda.LongTensor)
        r = triplet[1].type(torch.cuda.LongTensor)
        o = triplet[2].type(torch.cuda.LongTensor)
        # print(s_hist)
        # print(rel_s_hist)
        ob_pred = self.predict_single(triplet, s_hist, rel_s_hist, att_s_hist,
                                      self_att_s_hist, o_hist, rel_o_hist,
                                      att_o_hist, self_att_o_hist)
        o_label = o
        s_label = s
        # sub_pred = torch.sigmoid(sub_pred)
        ob_pred = torch.sigmoid(ob_pred)

        ground = ob_pred[o].clone()

        s_id = torch.nonzero(
            all_triplets[:, 0].type(torch.cuda.LongTensor) == s).view(-1)
        idx = torch.nonzero(
            all_triplets[s_id, 1].type(torch.cuda.LongTensor) == r).view(-1)
        idx = s_id[idx]
        idx = all_triplets[idx, 2].type(torch.cuda.LongTensor)
        ob_pred[idx] = 0
        ob_pred[o_label] = ground

        ob_pred_comp1 = (ob_pred > ground).data.cpu().numpy()
        ob_pred_comp2 = (ob_pred == ground).data.cpu().numpy()
        rank_ob = np.sum(ob_pred_comp1) + (
            (np.sum(ob_pred_comp2) - 1.0) / 2) + 1

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
