import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 2 to use

import torch
import torch.nn as nn

from errinj.errorutils import error_gen_with_bias_fast
from errinj.modelutils import get_layers, get_model, find_sublayers
from errinj.evalutils import evaluate_perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer

def replace_linear_with_bit_error(
    model,
    bit_error_rate=1e-4,
    wbits=32,
    device="cuda"
):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None

            new_module = LinearWithBitError(
                in_features, out_features, bias,
                bit_error_rate,
                wbits,
                device
            )
            new_module.weight.data = module.weight.data.clone()
            if bias:
                new_module.bias.data = module.bias.data.clone()
            
            setattr(model, name, new_module)
        
        elif isinstance(module, nn.Module):
            replace_linear_with_bit_error(module, bit_error_rate, wbits, device)
    
    return model

class LinearWithBitError(nn.Linear):
    def __init__(
            self, 
            in_features, 
            out_features, 
            bias=True, 
            bit_error_rate=1e-4,
            wbits=32,
            device="cuda",
        ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.bit_error_rate = bit_error_rate
        self.wbits = wbits
        self.device = device
        self.row_bias = None
        self.col_bias = None
    
    def inject_bit_error(self, weight, seed):
        with torch.no_grad():
            err_mat = error_gen_with_bias_fast(
                weight, 
                self.bit_error_rate, 
                seed, 
                self.wbits,
                row_bias=self.row_bias,
                col_bias=self.col_bias,
            ).to(self.device)
            corrupted_weight = (weight.to(torch.int32) ^ err_mat).to(weight.dtype)
        return corrupted_weight
    
    def forward(self, input, seed):
        corrupted_weight = self.inject_bit_error(self.weight, seed)
        return nn.functional.linear(input, corrupted_weight, self.bias)

class LayersWithBitError(nn.Module):
    def __init__(
        self,
        layers,
        start_seed=42,
        bit_error_rate=1e-4,
        wbits=32,
        device="cuda"            
    ):
        super().__init__()
        self.layers = layers
        self.start_seed = start_seed
        self.bit_error_rate = bit_error_rate
        self.wbits = wbits
        self.device = device
        self.token_step = 0

    def set_token_step(self, step):
        self.token_step = step
    
    def forward(self, x):
        # set seed dynamically        
        current_seed = self.start_seed + self.token_step
        # layer-wise seed interval
        layer_seed_offset = 100

        for layer_num, layer in enumerate(self.layers):
            for sublayer_num, sublayer in enumerate(layer):
                if isinstance(sublayer, LinearWithBitError):
                    sublayer_seed = current_seed + layer_seed_offset * layer_num + sublayer_num
                    x = sublayer(x, seed=sublayer_seed)
        
        self.token_step += 1
        return x


def main():
    model_id = '/raid/LLM/opt-125m'
    is_quantized = False
    model_dtype = "auto"
    model = get_model(model_id, is_quantized, dtype=model_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    device = "cuda"
    bit_error_rate = 1e-4
    wbits = 4
    model = replace_linear_with_bit_error(
        model,
        bit_error_rate=bit_error_rate,
        wbits=wbits,
        device=device,
    )

    layers = get_layers(model)
    layers = LayersWithBitError(layers, 42, bit_error_rate, wbits, device)

    evaluate_perplexity(model, tokenizer, nsample=128)

if __name__ == '__main__':
    main()