import torch
from bit2bcc import b2c

input1 = torch.rand([10, 2])


def c2b(input_tensor, block_len):
    output = torch.zeros([input_tensor.shape[0], block_len * input_tensor.shape[1]])
    for i in range(input_tensor.shape[0]):
        for j in range(input_tensor.shape[1]):
            num = input_tensor[i][j]
            binary_string = bin(num)[2:].zfill(32)
            binary_list = [int(bit) for bit in binary_string]
            output[i, j:j + 32] = torch.tensor(binary_list)
    return output

mat  = torch.randint(0, 2, (10, 32))
mat1=b2c(mat,32)
print(torch.equal(mat,c2b(mat1,32)))