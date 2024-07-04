import torch


def gemm_based_search(mat):
    redun = torch.mm(mat, mat.T).fill_diagonal_(0)
    redun[redun < 2] = 0
    indice1 = torch.where(redun != 0)
    red_mask = mat[indice1[0]] * mat[indice1[1]]
    unique = torch.unique(red_mask, dim=0, return_inverse=True)[0]
    return unique


def bcc_mat_mul(in1, in2):
    out = torch.zeros([in1.shape[0], in2.shape[0],in1.shape[0]])
    for i in range(in1.shape[0]):
        for j in range(i + 1, in2.shape[0]):
            temp = in1[i] & in2[j]
            popcount = 0
            for k in range(temp.shape[0]):
                popcount += bin(temp[k]).count('1')
            if popcount>1:
                out[i][j]=temp
    return out



