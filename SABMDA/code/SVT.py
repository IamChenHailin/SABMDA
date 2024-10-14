import numpy as np
import torch
from scipy.linalg import svd
from scipy.stats import mode

def generate_mask(relation_matrix):
    a_mask = torch.ones((134, 134), dtype=torch.float).to(device='cuda:0')
    b_mask = torch.ones((1177, 1177), dtype=torch.float).to(device='cuda:0')

    # Convert relation_matrix to torch tensor if it's not already
    if not isinstance(relation_matrix, torch.Tensor):
        relation_matrix = torch.tensor(relation_matrix, dtype=torch.float).to(device='cuda:0')

    temp_mask = torch.cat((a_mask, relation_matrix), dim=1)
    temp1_mask = torch.cat((relation_matrix.transpose(0, 1), b_mask), dim=1)
    matrix_mask = torch.cat((temp_mask, temp1_mask), dim=0)

    return matrix_mask

def generate_mask1(relation_matrix):
    a_mask = torch.ones((39, 39), dtype=torch.float).to(device='cuda:0')
    b_mask = torch.ones((292, 292), dtype=torch.float).to(device='cuda:0')

    # Convert relation_matrix to torch tensor if it's not already
    if not isinstance(relation_matrix, torch.Tensor):
        relation_matrix = torch.tensor(relation_matrix, dtype=torch.float).to(device='cuda:0')

    temp_mask = torch.cat((a_mask, relation_matrix), dim=1)
    temp1_mask = torch.cat((relation_matrix.transpose(0, 1), b_mask), dim=1)
    matrix_mask = torch.cat((temp_mask, temp1_mask), dim=0)

    return matrix_mask


def svt(M, mask, max_iterations):
    tau = 10
    delta = 0.1
    Y = np.zeros_like(M)

    for k in range(max_iterations):
        try:
            U, S, Vh = svd(Y, full_matrices=False)
        except np.linalg.LinAlgError as e:
            print(f"SVD did not converge at iteration {k+1}. Error: {e}")
            break

        S = np.maximum(S - tau, 0)

        X = np.linalg.multi_dot([U, np.diag(S), Vh])
        Y += delta * mask * (M - X)

        recon_error = np.linalg.norm(mask * (X - M)) / np.linalg.norm(mask * M)
        print("[%d/%d] reconstruction_error : %.4f" % (k+1, max_iterations, recon_error))

    return X

def constructHNet(drug_dis_matrix, drug_matrix, dis_matrix):

    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    return np.vstack((mat1, mat2))

def majority_vote(scores1, scores2):
    """
    通过多数投票的方法将两个得分矩阵合并成一个最终的分类结果。

    参数:
    scores1: 第一个得分矩阵
    scores2: 第二个得分矩阵

    返回:
    最终的分类结果矩阵
    """
    # 获取每个得分矩阵中每个样本的最高得分类别
    preds1 = torch.argmax(scores1, dim=1)
    preds2 = torch.argmax(scores2, dim=1)

    # 合并预测结果
    preds = torch.stack([preds1, preds2])

    # 通过多数投票得到最终的分类结果
    final_preds, _ = mode(preds.cpu().numpy(), axis=0)
    # 将结果转换回torch.tensor，并调整形状
    final_preds = torch.tensor(final_preds.flatten())

    # 将结果扩展回原始维度
    final_scores = torch.zeros_like(scores1)
    for i in range(final_preds.size(0)):
        final_scores[i, final_preds[i]] = 1

    return final_scores