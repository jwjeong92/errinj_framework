import torch
import torch.nn as nn
from errinj.errorutils import error_gen_with_bias_fast, error_injection

class LinearWithBitError(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bit_error_rate=1e-4, wbits=32):
        super().__init__(in_features, out_features, bias)
        self.bit_error_rate = bit_error_rate
        self.wbits = wbits
        self.current_seed = None  # 현재 seed를 저장

    def inject_bit_error(self, weight, seed):
        # Error Injection with Bias
        error_matrix = error_gen_with_bias_fast(
            weight, 
            rate=self.bit_error_rate, 
            seed=seed, 
            wbits=self.wbits
        ).to(weight.device)
        corrupted_weight = (weight.to(torch.int32) ^ error_matrix).to(weight.dtype)
        return corrupted_weight

    def forward(self, input, seed=None):
        # Seed 기본값 설정
        seed = seed if seed is not None else self.current_seed
        if seed is None:
            raise ValueError("Seed must be provided either through 'forward()' or 'current_seed'.")
        # Weight에 Bit Error 주입
        corrupted_weight = self.inject_bit_error(self.weight.clone(), seed)
        return nn.functional.linear(input, corrupted_weight, self.bias)