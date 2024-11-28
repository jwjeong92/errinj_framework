import scipy.sparse as sp
import torch
import numpy as np

def conv_bin2int(bin, wbits):
    bin = bin.reshape(wbits, -1)
    sum = torch.zeros(bin.size(-1), dtype=torch.int32)
    for i in range(len(bin)):
        sum += bin[i] * 2**((wbits-1)-i)
    return sum
    
def error_gen(param, rate, seed, wbits):
    orig_size = param.size()
    bitwidth = param.data.element_size()*8
    
    bin_error = torch.tensor(sp.random(np.prod(orig_size), wbits, density=rate, dtype=bool, random_state=np.random.default_rng(seed)).toarray())
    error_matrix = conv_bin2int(bin_error, wbits)
    del bin_error
    return error_matrix.view(orig_size)

def error_injection(param, rate, seed, wbits, device="cuda"):
    err_mat = error_gen(param, rate, seed, wbits).to(device)
    int_form = err_mat.dtype
    if param.element_size() == 2:
        return err_mat.to(torch.int16)
    elif param.element_size() == 1:
        return err_mat.to(torch.int8)
    else:
        return err_mat.to(torch.int32)
    

def error_gen_with_bias_fast(param, rate, seed, wbits, row_bias=None, col_bias=None):
    '''
    Generate the errors to bias in speicific col/row

    Args:
        param (torch.Tensor): origianl weight tensor
        rate (float): rate injecting bit error
        seed (int): random seed
        wbits (int): bit-width of weights
        row_bias (np.ndarray): row-wise error occurance weight
        col_bias (np.ndarray): col-wise error occurance weight

    Returns:
        torch.Tensor: corrupted weight tensor
    '''
    torch.manual_seed(seed)
    rows, cols = param.size()
    num_elements = rows * cols
    total_bits = num_elements * wbits
    num_errors = int(rate * total_bits)

    if row_bias is not None:
        row_cdf = np.cumsum(row_bias / np.sum(row_bias))
    else:
        row_cdf = None

    if col_bias is not None:
        col_cdf = np.cumsum(col_bias / np.sum(col_bias))    
    else:
        col_cdf = None
    
    if row_cdf is not None:
        row_indices = np.searchsorted(
            row_cdf, np.random.rand(num_errors // wbits)
        )
    else:
        row_indices = np.random.randint(
            0, rows, num_errors // wbits
        )
    
    if col_cdf is not None:
        col_indices = np.searchsorted(
            col_cdf, np.random.rand(num_errors // wbits)
        )
    else:
        col_indices = np.random.randint(
            0, cols, num_errors // wbits
        )
    bit_positions = np.random.randint(0, wbits, num_errors)

    error_mask = torch.zeros(num_elements, dtype=torch.int32, device=param.device) 
    flattened_indices = torch.tensor(row_indices * cols + col_indices, device=param.device)
    for idx, bit in zip(flattened_indices, bit_positions):
        error_mask[idx] ^= (1 << (wbits - 1 - bit))

    return error_mask.view(param.size())

if __name__ == '__main__':
    a = torch.randn(10,10)
    tensor = error_gen_with_bias_fast(
        param=a,
        rate = 0.1,
        seed=42,
        wbits=4,
        row_bias=np.array([1.0 if i < a.size(0)//2 else 0.1 for i in range(a.shape[0])]),
        col_bias=np.array([1.0 if i % 2 == 0 else 0.1 for i in range(a.shape[1])])

    )
    print(tensor)