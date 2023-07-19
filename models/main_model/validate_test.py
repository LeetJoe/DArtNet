import argparse
import numpy as np
import torch
import utils_test as utils
import os
from model_test import DArtNet
import pickle
import collections
import time

result = collections.namedtuple(
    "result",
    ["epoch", "MRR", "sub_att_loss", "MR", "Hits1", "Hits3", "Hits10"])

result_dict = {}


# 与 test.py 文件里的 test 方法除了读取的 txt 文件有所不同以外，其它完全一样。
def test(args):
    # load data todo 这些数据需要重复加载吗？应该可以在外层加载。
    num_nodes, num_rels = utils.get_total_number(args.dataset_path, 'stat.txt')
    test_data, test_times = utils.load_hexaruples(args.dataset_path,
                                                  'valid.txt')
    total_data, total_times = utils.load_hexaruples(args.dataset_path,
                                                    'train.txt', 'valid.txt')

    model_dir = 'models/' + args.dataset + '/{}-{}-{}-{}'.format(
        args.dropout, args.n_hidden, args.gamma, args.num_k)
    model_state_file = model_dir + '/epoch-{}.pth'.format(args.epoch)

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed_all(999)

    model = DArtNet(num_nodes,
                    args.n_hidden,
                    num_rels,
                    model=args.model,
                    seq_len=args.seq_len,
                    num_k=args.num_k,
                    gamma=args.gamma)

    if use_cuda:
        model.cuda()

    # todo 这些数据也都可以在外层加载，没必要每次循环里重复加载。
    test_sub_entity = '/valid_entity_s_history_data.txt'
    test_sub_rel = '/valid_rel_s_history_data.txt'
    test_sub_att = '/valid_att_s_history_data.txt'
    test_sub_self_att = '/valid_self_att_s_history_data.txt'

    test_ob_entity = '/valid_entity_o_history_data.txt'
    test_ob_rel = '/valid_rel_o_history_data.txt'
    test_ob_att = '/valid_att_o_history_data.txt'
    test_ob_self_att = '/valid_self_att_o_history_data.txt'

    with open(args.dataset_path + test_sub_entity, 'rb') as f:
        entity_s_history_data_test = pickle.load(f)
    with open(args.dataset_path + test_sub_rel, 'rb') as f:
        rel_s_history_data_test = pickle.load(f)
    with open(args.dataset_path + test_sub_att, 'rb') as f:
        att_s_history_data_test = pickle.load(f)
    with open(args.dataset_path + test_sub_self_att, 'rb') as f:
        self_att_s_history_data_test = pickle.load(f)

    with open(args.dataset_path + test_ob_entity, 'rb') as f:
        entity_o_history_data_test = pickle.load(f)
    with open(args.dataset_path + test_ob_rel, 'rb') as f:
        rel_o_history_data_test = pickle.load(f)
    with open(args.dataset_path + test_ob_att, 'rb') as f:
        att_o_history_data_test = pickle.load(f)
    with open(args.dataset_path + test_ob_self_att, 'rb') as f:
        self_att_o_history_data_test = pickle.load(f)

    print(f'\nstart testing model file : {model_state_file}')

    checkpoint = torch.load(model_state_file,
                            map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    # 将 model.xxx_his_test/cache 里的数据初始化为空串。
    model.init_history()

    model.latest_time = checkpoint['latest_time']

    print("Using epoch: {}".format(checkpoint['epoch']))

    # total data 是 train.txt + valid.txt
    total_data = torch.from_numpy(total_data)
    # test_data 是 valid.txt
    test_data = torch.from_numpy(test_data)

    # 设置 model 为 evaluation 模式，实际上执行的是 model.train(False)，即非训练模式。
    model.eval()
    total_att_sub_loss = 0
    total_ranks = np.array([])
    total_ranks_filter = np.array([])
    ranks = []

    # torch.no_grad()是一个上下文管理函数，告诉torch不要进行梯度计算。在推理的时候，如果确定不会调用 tensor.backward() ,可以使用此函数来降低内存使用。
    # 其实际效果是设置了一个参数值 requires_grad=False。
    with torch.no_grad():
        latest_time = test_times[0]     # validate 脚本里的所有 test 数据实际上都是 validate.txt 里的数据，并非指 test.txt 里的数据
        j = 0
        while j < len(test_data):  # 遍历 validate data
            k = j # k >= j, 从 j 开始，不超过 len(test_data)
            while k < len(test_data):
                # test_data[x][-1] 是 time
                if test_data[k][-1] == test_data[j][-1]:
                    k += 1
                else:
                    break

            # todo 从 test_data[j] 到 test_data[k] 之间的项有相同的 time
            start = j
            while start < k: # todo start 到 k 之间可能隔了多个 batch_size，每"最多batch_size"项数据进行一次预测。
                end = min(k, start + args.batch_size)

                batch_data = test_data[start:end].clone()
                s_hist = entity_s_history_data_test[start:end].copy()
                o_hist = entity_o_history_data_test[start:end].copy()
                rel_s_hist = rel_s_history_data_test[start:end].copy()
                rel_o_hist = rel_o_history_data_test[start:end].copy()
                att_s_hist = att_s_history_data_test[start:end].copy()
                att_o_hist = att_o_history_data_test[start:end].copy()
                self_att_s_hist = self_att_s_history_data_test[start:end].copy(
                )
                self_att_o_hist = self_att_o_history_data_test[start:end].copy(
                )

                if use_cuda:
                    batch_data = batch_data.cuda()

                # 对 batch_data 进行预测， xxx_hist 是与 batch_data 的对应的关系数据，通过 gen_history.py 生成的那些。
                # 返回的 loss_sub 其实是 loss_att_sub，是对 att 的预测。
                loss_sub = model.predict(batch_data, s_hist, rel_s_hist,
                                         att_s_hist, self_att_s_hist, o_hist,
                                         rel_o_hist, att_o_hist,
                                         self_att_o_hist)

                # 将 loss_sub 乘上 batch_size(这一批次的数量，可能小于参数 batch-size)，得到总体 loss。
                # todo 所以 loss_sub 原本是按 batch_size 平均过？必须乘回去得到原本的 total loss？
                total_att_sub_loss += (loss_sub.item() * (end - start + 1))

                start += args.batch_size

            # 从 j 到 k，即前面同一 time 的所有数据，上一个循环使用 batch_size 进行了分批，这次就是未分批的全部，用 i 对它进行遍历。
            for i in range(j, k):
                batch_data = test_data[i].clone()
                s_hist = entity_s_history_data_test[i].copy()
                o_hist = entity_o_history_data_test[i].copy()
                rel_s_hist = rel_s_history_data_test[i].copy()
                rel_o_hist = rel_o_history_data_test[i].copy()
                att_s_hist = att_s_history_data_test[i].copy()
                att_o_hist = att_o_history_data_test[i].copy()
                self_att_s_hist = self_att_s_history_data_test[i].copy()
                self_att_o_hist = self_att_o_history_data_test[i].copy()

                if use_cuda:
                    batch_data = batch_data.cuda()

                # batch_data 在这里是一条单一的 6 元组，后面的数据都是跟它相对应的关系信息。
                # 最后的 total_data 是 train data + validate data
                # ranks_pred 的评价的是，对 tail 的预测里，那些实际并不存在的 [h,r,o_i] 关系，但是对 tail 的预测概率却比对应的原 [h,r,o]
                # 的概率要大的 tails 的数量. 比如 node id 为1~10, 对[1,2,3]，预测结果显示 p(3)=0.5, 原始数据里存在 [1,2,1],[1,2,2]
                # 这两种关系，那就先把 1,2 排除，3 自身也要排除；然后在 4~10 里，如果有 p(4)>=p(3),p(6)>=p(3),p(8)>=p(3),那么 rank_pred
                # 的值与 3 线性相关。
                ranks_pred = model.evaluate_filter(batch_data, s_hist,
                                                   rel_s_hist, att_s_hist,
                                                   self_att_s_hist, o_hist,
                                                   rel_o_hist, att_o_hist,
                                                   self_att_o_hist, total_data)

                total_ranks_filter = np.concatenate(
                    (total_ranks_filter, ranks_pred))

            j = k

    # 这里得到的 total_ranks_filter 长度为 252,其长度与 test_data 相同，每一条数据都有一个rank_pred.
    ranks.append(total_ranks_filter)

    for rank in ranks:
        total_ranks = np.concatenate((total_ranks, rank))
    # todo 前面这几步在搞什么！？把 total_ranks_filter 放进 ranks,又拿出来 concatinate，得到的 total_ranks 与 total_ranks_filter 完全就是一个东西。

    # 这里的 mrr 含义是求 total_ranks 里所有元素的倒数的和的平均值
    mrr = np.mean(1.0 / total_ranks)
    # 这里的 mr 就是 total_ranks 里所有元素的和的平均值
    mr = np.mean(total_ranks)
    hits = []

    # 所以这里 hit 的含义实际上是指，那些概率大于或等于 ground 的位置，都被视为是一次命中。
    # todo 但这个命中是错误的，因为实际中并没有这样的关系，所以这个命中应该越少越好？
    for hit in [1, 3, 10]:
        # 这里的 np.mean 的含义是：对 total_ranks 里值 <= hit 的元素的"个数"进行累加然后除以 len(total_ranks)
        avg_count = np.mean((total_ranks <= hit))
        hits.append(avg_count)

        print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count))
    print("MRR (filtered): {:.6f}".format(mrr))
    print("MR (filtered): {:.6f}".format(mr))
    print("test att sub Loss: {:.6f}".format(total_att_sub_loss /
                                             (len(test_data))))

    result_epoch = result(epoch=args.epoch,
                          MRR=100 * mrr,
                          sub_att_loss=total_att_sub_loss / len(test_data),
                          MR=mr,
                          Hits1=100 * hits[0],
                          Hits3=100 * hits[1],
                          Hits10=100 * hits[2])

    result_dict[args.epoch] = result_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DArtNet')
    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        help="dropout probability")
    parser.add_argument("-d",
                        "--dataset_path",
                        type=str,
                        help="dataset_path to use")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--model", type=int, default=0)
    parser.add_argument("--n-hidden",
                        type=int,
                        default=200,
                        help="number of hidden units")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--num-k",
                        type=int,
                        default=10,
                        help="cuttoff position")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--no-wait", action='store_true')

    args = parser.parse_args()

    model_dir = 'models/' + args.dataset + '/{}-{}-{}-{}'.format(
        args.dropout, args.n_hidden, args.gamma, args.num_k)

    try:
        # 如果已存在生成文件，则读取其中的内容
        with open(model_dir + '/compiled_results_validate.tsv', 'r') as f:
            res = f.readlines()[1:]

        for r in res:
            continue
            a = r.strip().split('\t')
            result_dict[int(a[0])] = result(epoch=int(a[0]),
                                            MRR=float(a[1]),
                                            sub_att_loss=float(a[5]),
                                            MR=float(a[6]),
                                            Hits1=float(a[2]),
                                            Hits3=float(a[3]),
                                            Hits10=float(a[4]))
    except FileNotFoundError as _:
        pass

    try:
        while True:
            flag = False
            # 扫描所有的对应 model 目录里的 epoch-i.pth 文件，对文件名中的 i 排序并以此为遍历序列。
            for epoch in sorted([
                    int(file[6:-4]) for file in os.listdir(model_dir)
                    if (file[-4:] == '.pth' and file[:5] == 'epoch')
            ],
                                reverse=True):

                # 只有发现了有新的 epoch-i.pth，即 epoch=i 的记录在result_dict中不存在时，执行 test，结果汇总在 result_dict 变量中.
                if epoch not in result_dict:
                    args.epoch = epoch
                    print(f'testing epoch {epoch}')
                    model_state_file = model_dir + '/epoch-{}.pth'.format(
                        args.epoch)
                    flag = True
                    test(args)   # 全都在这里完成

                    with open(model_dir + '/compiled_results_validate.tsv',
                              'w') as f:
                        f.write(
                            'Epoch\tMRR\tHits1\tHits3\tHits10\tAttribute_Loss\tMR\n'
                        )
                        for key, val in result_dict.items():

                            f.write(
                                f'{key}\t{val.MRR}\t{val.Hits1}\t{val.Hits3}\t{val.Hits10}\t{val.sub_att_loss}\t{val.MR}\n'
                            )

                    break

            if not flag: # 如果前面执行了 test，则不等待，重新扫描目录；若扫描未发现新的 epoch-i.pth 文件，则进入 sleep。
                if args.no_wait:
                    break
                time.sleep(60)
    except KeyboardInterrupt as _: # 如果发生了手动打断，则在退出前把 result_dict 存入文件中。
        with open(model_dir + '/compiled_results_validate.tsv', 'w') as f:
            f.write('Epoch\tMRR\tHits1\tHits3\tHits10\tAttribute_Loss\tMR\n')
            for key, val in result_dict.items():

                f.write(
                    f'{key}\t{val.MRR}\t{val.Hits1}\t{val.Hits3}\t{val.Hits10}\t{val.sub_att_loss}\t{val.MR}\n'
                )
