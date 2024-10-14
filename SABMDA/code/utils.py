import torch
import numpy as np

def bnnr(alpha, beta, T, trIndex, tol1, tol2, maxiter, a, b):
    X = T.clone().to(device='cuda:0')
    W = X.clone().to(device='cuda:0')
    Y = X.clone().to(device='cuda:0')

    i = 1
    stop1 = 1
    stop2 = 1
    while stop1 > tol1 or stop2 > tol2:
        # the process of computing W
        tran = ((1/beta) * (Y + alpha * (T * trIndex)) + X).to(device='cuda:0')
        W = (tran - (alpha / (alpha + beta)) * (tran * trIndex)).to(device='cuda:0')
        W = (torch.clamp(W, min=a, max=b)).to(device='cuda:0')

        # the process of computing X
        u, s, v = torch.svd(W - 1/beta * Y)
        s = (torch.nn.functional.softshrink(s, 1/beta)).to(device='cuda:0')
        X_1 = (torch.matmul(torch.matmul(u, torch.diag(s)), v.t())).to(device='cuda:0')

        # the process of computing Y
        Y = (Y + beta * (X_1 - W)).to(device='cuda:0')

        stop1_0 = stop1
        stop1 = (torch.norm(X_1 - X, p='fro') / torch.norm(X, p='fro')).to(device='cuda:0')
        stop2 = (abs(stop1 - stop1_0) / max(1, abs(stop1_0))).to(device='cuda:0')

        X = X_1
        i += 1

        if i < maxiter:
            iter_ = i - 1
        else:
            iter_ = maxiter
            print('reach maximum iteration~~do not converge!!!')
            break

    T_recovery = W
    return T_recovery, iter_

def GIPSim(interaction, gamadd, gamall):
    nd, nl = interaction.size()

    # Calculate gamad for Gaussian kernel calculation
    sd = torch.norm(interaction, dim=1)**2
    gamad = nd / torch.sum(sd) * gamadd

    # Calculate gamal for Gaussian kernel calculation
    sl = torch.norm(interaction, dim=0)**2
    gamal = nl / torch.sum(sl) * gamall

    # Calculate Gaussian kernel for the similarity between disease: kd
    kd = torch.zeros(nd, nd)
    for i in range(nd):
        for j in range(nd):
            kd[i, j] = torch.exp(-gamad * torch.norm(interaction[i, :] - interaction[j, :])**2)

    # Calculate Gaussian kernel for the similarity between microbe: kl
    kl = torch.zeros(nl, nl)
    for i in range(nl):
        for j in range(nl):
            kl[i, j] = torch.exp(-gamal * torch.norm(interaction[:, i] - interaction[:, j])**2)

    return kd, kl


def BNNR(Wrr, Wdr, Wdd, alpha, beta, tol1, tol2, maxiter, a, b):
    dn, dr = Wdr.size()
    # diseasesim, microbesim = GIPSim(Wdr, 1, 1)
    # Wdd = ((torch.tensor(Wdd) + diseasesim) / 2.0).to(device='cuda:0')
    # Wrr = ((torch.tensor(Wrr) + microbesim) / 2.0).to(device='cuda:0')
    T = torch.cat((torch.cat((torch.tensor(Wrr).to(device='cuda:0'), Wdr.t()), dim=1), torch.cat((Wdr, torch.tensor(Wdd).to(device='cuda:0')), dim=1)), dim=0).to(device='cuda:0')
    t1, _ = T.size()
    trIndex = (T != 0).double().to(device='cuda:0')
    WW, _ = bnnr(alpha, beta, T, trIndex, tol1, tol2, maxiter, a, b)
    M_recovery = (WW[(t1-dn):t1, :dr]).to(device='cuda:0')
    return M_recovery


def BNNR1(Wrr, alpha, beta, tol1, tol2, maxiter, a, b):
    dn = Wrr.size
    # diseasesim, microbesim = GIPSim(Wdr, 1, 1)
    # Wdd = ((torch.tensor(Wdd) + diseasesim) / 2.0).to(device='cuda:0')
    # Wrr = ((torch.tensor(Wrr) + microbesim) / 2.0).to(device='cuda:0')
    T = torch.tensor(Wrr).to(device='cuda:0')
    t1, _ = T.size()
    trIndex = (T != 0).double().to(device='cuda:0')
    WW, _ = bnnr(alpha, beta, T, trIndex, tol1, tol2, maxiter, a, b)
    M_recovery = (WW[(t1-dn):t1, :]).to(device='cuda:0')
    return M_recovery

def calculate_loss(pred, pos_edge_idx, neg_edge_idx, device):
    pos_pred_socres = pred[pos_edge_idx[0], pos_edge_idx[1]]
    neg_pred_socres = pred[neg_edge_idx[0], neg_edge_idx[1]]
    pred_scores = torch.hstack((pos_pred_socres, neg_pred_socres)).to(device)
    true_labels = torch.hstack((torch.ones(pos_pred_socres.shape[0]), torch.zeros(neg_pred_socres.shape[0]))).to(device)
    loss_fun = torch.nn.BCELoss(reduction='mean').to(device)
    # loss_fun=torch.nn.BCEWithLogitsLoss(reduction='mean')
    return loss_fun(pred_scores.float(), true_labels.float()).to(device)

def calculate_evaluation_metrics(pred_mat, pos_edges, neg_edges, i):
    pos_pred_socres = pred_mat[pos_edges[0], pos_edges[1]]
    neg_pred_socres = pred_mat[neg_edges[0], neg_edges[1]]
    pred_labels = np.hstack((pos_pred_socres, neg_pred_socres))
    true_labels = np.hstack((np.ones(pos_pred_socres.shape[0]), np.zeros(neg_pred_socres.shape[0])))
    # np.savetxt(f'pred_labels{i}.txt', pred_labels)
    # np.savetxt(f'true_labels{i}.txt', true_labels)
    return get_metrics1(true_labels, pred_labels)

def calculate_evaluation_metrics1(pred_mat, pos_edges, neg_edges, i):
    pos_pred_socres = pred_mat[pos_edges[0], pos_edges[1]]
    neg_pred_socres = pred_mat[neg_edges[0], neg_edges[1]]
    pred_labels = np.hstack((pos_pred_socres, neg_pred_socres))
    true_labels = np.hstack((np.ones(pos_pred_socres.shape[0]), np.zeros(neg_pred_socres.shape[0])))
    # np.savetxt(f'pred_labels{i}.txt', pred_labels)
    # np.savetxt(f'true_labels{i}.txt', true_labels)
    return get_metrics1(true_labels, pred_labels)

def get_metrics1(real_score, predict_score):
    real_score, predict_score = real_score.flatten(), predict_score.flatten()
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    # thresholds = sorted_predict_score[range(
    #     sorted_predict_score_num )]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]

    # np.savetxt(roc_path.format(i), ROC_dot_matrix)

    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]

    # np.savetxt(pr_path.format(i), PR_dot_matrix)

    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)
    # plt.plot(x_ROC, y_ROC)
    # plt.plot(x_PR, y_PR)
    # plt.show()
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    print( ' auc:{:.4f} ,aupr:{:.4f},f1_score:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, specificity:{:.4f}, precision:{:.4f}'.format(auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision))
    return [auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision]

def torch_corr_x_y(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """
    :param tensor1: a matrix, torch Tensor
    :param tensor2: a matrix, torch Tensor
    :return: corr(tensor1, tensor2)
    """
    assert tensor1.size()[1] == tensor2.size()[1], "Different size!"
    tensor2 = torch.t(tensor2)
    mean1 = torch.mean(tensor1, dim=1).view([-1, 1])
    mean2 = torch.mean(tensor2, dim=0).view([1, -1])
    lxy = torch.mm(torch.sub(tensor1, mean1), torch.sub(tensor2, mean2))
    lxx = torch.diag(torch.mm(torch.sub(tensor1, mean1), torch.t(torch.sub(tensor1, mean1))))
    lyy = torch.diag(torch.mm(torch.t(torch.sub(tensor2, mean2)), torch.sub(tensor2, mean2)))
    std_x_y = torch.mm(torch.sqrt(lxx).view([-1, 1]), torch.sqrt(lyy).view([1, -1]))
    corr_x_y = torch.div(lxy, std_x_y)
    return corr_x_y

def scale_sigmoid(tensor: torch.Tensor, alpha: int or float):
    """
    :param tensor: a torch tensor, range is [-1, 1]
    :param alpha: an scale parameter to sigmod
    :return: mapping tensor to [0, 1]
    """
    alpha = torch.tensor(alpha, dtype=torch.float32, device=tensor.device)
    output = torch.sigmoid(torch.mul(alpha, tensor))
    return output

def constructNet(m_d_matrix,m_matrix,d_matrix):
    mat1 = np.hstack((m_matrix, m_d_matrix))
    mat2 = np.hstack((m_d_matrix.T, d_matrix))
    adj = np.vstack((mat1, mat2))
    return adj

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)