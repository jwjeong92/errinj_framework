import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"  # Set the GPU 2 to use


from transformers import AutoModelForCausalLM, AutoTokenizer
from errinj.module import LinearWithBitError
from errinj.evalutils import evaluate_perplexity
from errinj.modelutils import get_layers
import torch
def replace_linear_with_custom(model, start_seed=42, bit_error_rate=1e-4, wbits=32, device="cuda"):
    """
    Hugging Face 모델의 nn.Linear를 LinearWithBitError로 교체.
    """
    # 1단계: 변경할 모듈을 먼저 수집
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            modules_to_replace.append((name, module))
    
    # 2단계: 수집한 모듈을 교체
    seed_offset = 0
    for name, module in modules_to_replace:
        # 기존 nn.Linear를 LinearWithBitError로 교체
        new_module = LinearWithBitError(
            module.in_features, 
            module.out_features, 
            bias=module.bias is not None, 
            bit_error_rate=bit_error_rate, 
            wbits=wbits
        )
        new_module.weight.data = module.weight.data.clone()
        if module.bias is not None:
            new_module.bias.data = module.bias.data.clone()

        # Hook 추가: 각 레이어의 seed 설정
        def inject_seeded_error(module, inputs):
            nonlocal seed_offset
            module.current_seed = start_seed + seed_offset  # 고유 seed 설정
            seed_offset += 1
        new_module.register_forward_pre_hook(inject_seeded_error)

        # 모델의 모듈 교체
        parent_module, attr_name = get_parent_module_and_attr(model, name)
        setattr(parent_module, attr_name, new_module)

    return model

def get_parent_module_and_attr(model, module_name):
    """
    주어진 모듈 이름에 따라 부모 모듈과 속성 이름을 반환.
    Args:
        model (nn.Module): 모델
        module_name (str): 모듈 이름 (e.g., "encoder.layers.0.attention")
    Returns:
        parent_module (nn.Module): 부모 모듈
        attr_name (str): 교체할 모듈의 속성 이름
    """
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]

# OPT 모델 로드
model_name = "/raid/LLM/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.half)
evaluate_perplexity(model.to("cuda"), tokenizer)
# 커스텀 Linear 교체
layers = get_layers(model)
layers = replace_linear_with_custom(layers, start_seed=42, bit_error_rate=1e-4, wbits=4, device="cuda")
model = model.to("cuda")

evaluate_perplexity(model, tokenizer)