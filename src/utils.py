import torch


def to_torch(np_array):
    return torch.tensor(np_array).type(torch.FloatTensor)


def to_numpy(torch_tensor):
    return torch_tensor.data.numpy()


def smart_init(lstm):
    """ortho init lstm with forget gate bias = 1

    Parameters
    ----------
    lstm : a pytorch lstm object
        Description of parameter `lstm`.

    Returns
    -------
    type
        "inited" lstm

    """
    for name, p in lstm.named_parameters():
        if 'weight' in name:
            torch.nn.init.orthogonal_(p)
        elif 'bias' in name:
            torch.nn.init.constant_(p, 0)
    # Set LSTM forget gate bias to 1
    for name, p in lstm.named_parameters():
        if 'bias' in name:
            n = p.size(0)
            forget_start_idx, forget_end_idx = n // 4, n // 2
            torch.nn.init.constant_(p[forget_start_idx:forget_end_idx], 1)
    return lstm
