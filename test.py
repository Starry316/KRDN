import math
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from time import time
from prettytable import PrettyTable
from torch.utils.data import Dataset, DataLoader, random_split
from utils.parser_test import parse_args
from utils.data_loader import load_data
from modules.KRDN import Recommender
from utils.evaluate import test
from utils.helper import early_stopping
import multiprocessing
from STTools.Torch import saveModel
from STTools.Utils import mkdir
from STTools.Logger import STLogger
cores = multiprocessing.cpu_count()
n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0

def get_neg_one(user):
    user = int(user)
    each_negs = list()
    neg_item = np.random.randint(low=0, high=n_items, size=args.num_neg_sample)
    if len(set(neg_item) & set(train_user_set[user])) == 0:
        each_negs += list(neg_item)
    else:
        neg_item = list(set(neg_item) - set(train_user_set[user]))
        each_negs += neg_item
        while len(each_negs) < args.num_neg_sample:
            n1 = np.random.randint(low=0, high=n_items, size=1)[0]
            if n1 not in train_user_set[user]:
                each_negs += [n1]
    return each_negs

def get_feed_data(train_entity_pairs, train_user_set):
    def negative_sampling(user_item, train_user_set):
        pool = multiprocessing.Pool(cores)
        neg_items = pool.map(get_neg_one, user_item.cpu().numpy()[:, 0])
        pool.close()
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,train_user_set))
    return feed_dict

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device, train_user_set
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph = load_data(args)

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    train_user_set = user_dict['train_user_set']

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    # test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    """define model"""
    model = Recommender(n_params, args, graph, train_cf_pairs).to(device)

    
    """load model"""
    model.load_state_dict(torch.load(args.load_path))


    """testing"""
    model.eval()
    test_s_t = time()
    STLogger.info('test start here')
    with torch.no_grad():
        ret = test(model, user_dict, n_params)
    test_e_t = time()
    STLogger.info('test done here')

    train_res = PrettyTable()
    train_res.field_names = ["tesing time", "recall", "ndcg", "precision", "hit_ratio"]
    train_res.add_row(
        [test_e_t - test_s_t, ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
    )
    print(train_res)
    f = open('./result/{}.txt'.format(args.dataset), 'a+')
    f.write(str(train_res) + '\n')
    f.close()
    STLogger.info('writing done here')
