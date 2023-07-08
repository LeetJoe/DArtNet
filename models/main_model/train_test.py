import argparse
import numpy as np
import time
import torch
import utils_test as utils
import os
from model_test import DArtNet
from sklearn.utils import shuffle
import pickle


def train(args):
    # load data
    num_nodes, num_rels = utils.get_total_number(args.dataset_path, 'stat.txt')
    train_data, train_times = utils.load_hexaruples(args.dataset_path,
                                                    'train.txt')

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    seed = 999
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # 按照参数 dropout, n_hidden, gama, num_k 存储和加载模型
    model_dir = 'models/' + args.dataset + '/{}-{}-{}-{}'.format(
        args.dropout, args.n_hidden, args.gamma, args.num_k)

    os.makedirs('models', exist_ok=True)
    os.makedirs('models/' + args.dataset, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("start training...")
    model = DArtNet(num_nodes,    # 1
                    args.n_hidden,    # 2
                    num_rels,    # 3
                    dropout=args.dropout,    # 4
                    model=args.model,    # 5
                    seq_len=args.seq_len,    # 6
                    num_k=args.num_k,    # 7
                    gamma=args.gamma)    # 8

    print('model initialized')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=0.00001)
    print('optimizer initialized')

    if use_cuda:
        model.cuda()

    train_sub_entity = '/train_entity_s_history_data.txt'
    train_sub_rel = '/train_rel_s_history_data.txt'
    train_sub_att = '/train_att_s_history_data.txt'
    train_sub_self_att = '/train_self_att_s_history_data.txt'

    train_ob_entity = '/train_entity_o_history_data.txt'
    train_ob_rel = '/train_rel_o_history_data.txt'
    train_ob_att = '/train_att_o_history_data.txt'
    train_ob_self_att = '/train_self_att_o_history_data.txt'

    with open(args.dataset_path + train_sub_entity, 'rb') as f:
        entity_s_history_data_train = pickle.load(f)
    with open(args.dataset_path + train_sub_rel, 'rb') as f:
        rel_s_history_data_train = pickle.load(f)
    with open(args.dataset_path + train_sub_att, 'rb') as f:
        att_s_history_data_train = pickle.load(f)
    with open(args.dataset_path + train_sub_self_att, 'rb') as f:
        self_att_s_history_data_train = pickle.load(f)

    with open(args.dataset_path + train_ob_entity, 'rb') as f:
        entity_o_history_data_train = pickle.load(f)
    with open(args.dataset_path + train_ob_rel, 'rb') as f:
        rel_o_history_data_train = pickle.load(f)
    with open(args.dataset_path + train_ob_att, 'rb') as f:
        att_o_history_data_train = pickle.load(f)
    with open(args.dataset_path + train_ob_self_att, 'rb') as f:
        self_att_o_history_data_train = pickle.load(f)

    entity_s_history_train = entity_s_history_data_train
    rel_s_history_train = rel_s_history_data_train
    att_s_history_train = att_s_history_data_train
    self_att_s_history_train = self_att_s_history_data_train

    entity_o_history_train = entity_o_history_data_train
    rel_o_history_train = rel_o_history_data_train
    att_o_history_train = att_o_history_data_train
    self_att_o_history_train = self_att_o_history_data_train

    print('data loaded')

    epoch = 0

    # retrain 可以加载之前训练的结果继续训练
    if args.retrain != 0:
        try:
            checkpoint = torch.load(model_dir + '/checkpoint.pth',
                                    map_location=f"cuda:{args.gpu}")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            model.latest_time = checkpoint['latest_time']
            model.to(torch.device(f"cuda:{args.gpu}"))
        except FileNotFoundError as _:
            try:
                e = sorted([
                    int(file[6:-4])
                    for file in os.listdir(model_dir) if file[-4:] == '.pth'
                ],
                           reverse=True)[0]
                checkpoint = torch.load(model_dir + '/epoch-{}.pth'.format(e),
                                        map_location=f"cuda:{args.gpu}")
                model.load_state_dict(checkpoint['state_dict'])
                epoch = checkpoint['epoch']
                model.latest_time = checkpoint['latest_time']
                model.to(torch.device(f"cuda:{args.gpu}"))
            except Exception as _:
                print('no model found')
                print('training from scratch')

    while True:
        print('a train loop started')
        model.train()   # todo 每一轮应该没有数据才会，为什么上来就 train？
        if epoch == args.max_epochs:
            break
        epoch += 1
        loss_epoch = 0
        loss_att_sub_epoch = 0
        # loss_att_ob_epoch = 0
        t0 = time.time()

        # shuffle 相当于把所有参数视为一个列向量组成一个矩阵，然后做行 shuffle。故所有参数的第一层维度大小必须相同，否则会报错。
        train_data, entity_s_history_train, rel_s_history_train, entity_o_history_train, rel_o_history_train, \
            att_s_history_train, self_att_s_history_train, att_o_history_train, self_att_o_history_train = shuffle(
            train_data, entity_s_history_train, rel_s_history_train,
            entity_o_history_train, rel_o_history_train, att_s_history_train,
            self_att_s_history_train, att_o_history_train,
            self_att_o_history_train)

        iteration = 0
        # 按 args.batch_size 把参数里的数组进行分组并按次序返回，用这种方式遍历
        for batch_data, s_hist, rel_s_hist, o_hist, rel_o_hist, att_s_hist, self_att_s_hist, att_o_hist, self_att_o_hist in utils.make_batch3(
                train_data, entity_s_history_train, rel_s_history_train,
                entity_o_history_train, rel_o_history_train,
                att_s_history_train, self_att_s_history_train,
                att_o_history_train, self_att_o_history_train,
                args.batch_size):
            iteration += 1
            print(f'iteration {iteration}', end='\r')
            batch_data = torch.from_numpy(batch_data)

            if use_cuda:
                batch_data = batch_data.cuda()

            '''
            print(len(batch_data))   # 1
            print(len(s_hist))   # 2
            print(len(rel_s_hist))   # 3
            print(len(att_s_hist))   # 4
            print(len(self_att_s_hist))   # 5
            print(len(o_hist))   # 6
            print(len(rel_o_hist))   # 7
            print(len(att_o_hist))   # 8
            print(len(self_att_o_hist))   # 9
            '''

            # todo 这里需要看 model 的具体实现
            loss, loss_att_sub = model.get_loss(
                batch_data,   # 1
                s_hist,   # 2
                rel_s_hist,   # 3
                att_s_hist,   # 4
                self_att_s_hist,   # 5
                o_hist,   # 6
                rel_o_hist,   # 7
                att_o_hist,   # 8
                self_att_o_hist)   # 9

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.grad_norm)  # clip gradients
            # todo Adam 也需要深入看一下，不然这里看不懂
            optimizer.step()
            optimizer.zero_grad()
            loss_epoch += loss.item()
            loss_att_sub_epoch += loss_att_sub.item()
            # loss_att_ob_epoch += loss_att_ob.item()

        t3 = time.time()
        print(
            "Epoch {:04d} | Loss {:.4f} | Loss_att_sub {:.4f} | time {:.4f} ".
            format(epoch, loss_epoch / (len(train_data) / args.batch_size),
                   loss_att_sub_epoch / (len(train_data) / args.batch_size),
                   t3 - t0))

        torch.save(
            {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'latest_time': model.latest_time,
            }, model_dir + '/epoch-{}.pth'.format(epoch))

        torch.save(
            {
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'latest_time': model.latest_time,
            }, model_dir + '/checkpoint.pth')

    print("training done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DArtNet')
    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        help="dropout probability")
    parser.add_argument("--n-hidden",
                        type=int,
                        default=200,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-d",
                        "--dataset_path",
                        type=str,
                        help="dataset_path to use")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--grad-norm",
                        type=float,
                        default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--max-epochs",
                        type=int,
                        default=20,
                        help="maximum epochs")
    parser.add_argument("--model", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--num-k",
                        type=int,
                        default=10,
                        help="cuttoff position")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--retrain", type=int, default=0)

    args = parser.parse_args()
    print(args)
    train(args)
