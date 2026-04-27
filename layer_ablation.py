import torch
import torch.nn as nn

class LayerAblationWrapper:
    """包装器：控制每层 LoRA 的开关"""
    
    def __init__(self, model, n_layers=12):
        self.model = model
        self.n_layers = n_layers
        self.ablated_layer = None
        self.original_forwards = {}
        
    def ablate_layer(self, layer_id):
        """消融指定层的 LoRA"""
        if layer_id is not None and (layer_id < 0 or layer_id >= self.n_layers):
            raise ValueError(f"Invalid layer_id: {layer_id}")
        
        # 恢复之前的消融
        self.restore()
        
        if layer_id is None:
            return  # 不消融任何层
        
        self.ablated_layer = layer_id
        
        # 找到该层的所有 LoRA 模块
        layer_prefix = f"decoder.sentence_encoder.layers.{layer_id}"
        
        for name, module in self.model.named_modules():
            if layer_prefix in name and 'lora' in name.lower():
                # 保存原始 forward
                if name not in self.original_forwards:
                    self.original_forwards[name] = module.forward
                
                # 替换为恒等映射（跳过 LoRA 分支）
                def identity_forward(x, *args, **kwargs):
                    # 如果是 LoraLinear，只返回主干输出
                    if hasattr(module, 'linear'):
                        return module.linear(x)
                    return x
                
                module.forward = identity_forward
    
    def restore(self):
        """恢复所有 LoRA 模块"""
        for name, original_forward in self.original_forwards.items():
            module = dict(self.model.named_modules())[name]
            module.forward = original_forward
        
        self.original_forwards.clear()
        self.ablated_layer = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.restore()
