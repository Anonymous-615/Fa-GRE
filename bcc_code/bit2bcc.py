

import torch


def pad_tensor(tensor):
    x = tensor.shape[0]
    # 计算 y，使其是 64 的倍数，并且大于等于 x
    y = ((x + 31) // 32) * 32

    # 创建一个新的 tensor，尺寸为 [x, y]，并用0填充
    padded_tensor = torch.zeros((x, y))

    # 将原始 tensor 的值复制到新 tensor 中
    padded_tensor[:, :x] = tensor

    return padded_tensor.int()





def b2c(input_tensor, block_len):
    """
    :param input_tensor:需要转码的布尔型数组
    :param block_len: 块长，根据位压缩后的数组要存储的数据格式决定，int32就是32位，long就是64位
    :return: 位压缩后的布尔型数组
    """

    output = torch.zeros([input_tensor.shape[0], input_tensor.shape[1] // block_len], dtype=torch.long)

    for i in range(input_tensor.shape[0]):
        for j in range(output.shape[1]):
            block = 0
            for k in range(block_len):
                block += input_tensor[i][j * block_len + k].item() * pow(2, block_len - k - 1)
            output[i][j] = block

    return output


