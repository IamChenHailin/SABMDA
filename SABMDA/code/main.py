import argparse
import copy
import os

import numpy as np
import torch

from SVT import *

from similarity_fusion import *
from model import *
from utils import calculate_loss, calculate_evaluation_metrics
# from contrast_loss import info_nce_loss
torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.cuda.empty_cache()


def set_seed(seed):
    torch.manual_seed(seed)
    #进行随机搜索的这个要注释掉
    # random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

set_seed(1)
repeat_times = 1

parser = argparse.ArgumentParser(description="Run CDGCN")
parser.add_argument('-device', type=str, default="cuda:0", help='cuda:number or cpu')
parser.add_argument('--lr', type=float, default=0.001,
                    help="the learning rate")
parser.add_argument('--wd', type=float, default=1e-5,
                    help="the weight decay for l2 normalizaton")
parser.add_argument('--layer_size', nargs='?', default=[128, 128],
                    help='Output sizes of every layer')
parser.add_argument('--alpha', type=float, default=1.0,
                    help="the scale for balance gcn and ni")
parser.add_argument('--beta', type=float, default=50.0,
                    help="the scale for sigmod")
parser.add_argument('--gamma', type=float, default=10,
                    help="the scale for sigmod")
parser.add_argument('--epochs', type=float, default=1,
                    help="the epochs for model")


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print(args.cuda)
hyperparam_dict = {
    'kfolds': 10,
}


MD = pd.read_csv("../Data/MD_A.csv", index_col=0)
MD_c = MD.copy()
MD_c.columns = range(0, MD.shape[1])
MD_c.index = range(0, MD.shape[0])
res = np.array(MD_c).T
k1 = 117
k2 = 13
m_fusion_sim, d_fusion_sim = get_fusion_sim(k1, k2)
m_fusion_sim = np.array(m_fusion_sim)
d_fusion_sim = np.array(d_fusion_sim)




pos_edges = np.loadtxt('pos1.txt').astype(int)
neg_edges = np.loadtxt('neg1.txt').astype(int)
idx = np.arange(pos_edges.shape[1])
np.random.shuffle(idx)
metrics_tensor = np.zeros((1, 7))
kfolds = hyperparam_dict['kfolds']
idx_splited = np.array_split(idx, kfolds)

for i in range(kfolds):
    tmp = []
    for j in range(1, kfolds):
        tmp.append(idx_splited[(j + i) % kfolds])
    training_pos_edges = np.loadtxt(f'trainpos{i}.txt').astype(int)
    training_neg_edges = np.loadtxt(f'trainneg{i}.txt').astype(int)
    test_pos_edges = np.loadtxt(f'testpos{i}.txt').astype(int)
    test_neg_edges = np.loadtxt(f'testneg{i}.txt').astype(int)
    temp_drug_dis = np.zeros((res.shape[0], res.shape[1]))
    temp_drug_dis[training_pos_edges[0], training_pos_edges[1]] = 1
    print(f'################Fold {i + 1} of {kfolds}################')
    print(i, ':after:', np.sum(temp_drug_dis.reshape(-1)))
    temp_drug_dis = torch.from_numpy(temp_drug_dis).to(torch.float32).to(device=args.device)
    mask = generate_mask(temp_drug_dis)
    Y = constructHNet(temp_drug_dis.cpu(), d_fusion_sim, m_fusion_sim)
    out1 = svt(np.array(Y), np.array(mask.cpu()), 500)
    out1 = torch.sigmoid(torch.tensor(out1))
    out1 = out1[:134, 134:]
    model = bnnr(m_fusion_sim, d_fusion_sim, out1.to(torch.float32).to(device=args.device), 100, args.alpha, args.beta, args.layer_size, args.device).to(device=args.device)
    for epoch in range(args.epochs):
        model.train()
        out = model().to(device=args.device)
        loss = calculate_loss(out, training_pos_edges, training_neg_edges, device=args.device)
        if (epoch + 1) % 100 == 0 or epoch == 0:
            pass
            print('------EPOCH {} of {}------'.format(epoch + 1, args.epochs))
            print('go on')

    model.eval()
    with torch.no_grad():
        pred_mat = model()
        pred_mat = pred_mat.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        pred_mat = torch.tensor(pred_mat).to(device='cuda:0')
        metrics = calculate_evaluation_metrics(np.array(pred_mat.cpu()), test_pos_edges, test_neg_edges, i)
        metrics_tensor += metrics
        del temp_drug_dis

print('Average result:', end='')
avg_metrics = metrics_tensor / kfolds
del metrics_tensor
print(avg_metrics)