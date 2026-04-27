import sys
sys.path.insert(0, ".")  # 确保用的是你项目里的 fairseq

import inspect
from fairseq import checkpoint_utils

# 关键：禁用 roberta 的 upgrade_state_dict_named（否则会访问 dense 导致报错）
import fairseq.models.roberta.model as roberta_model
def _no_upgrade(self, state_dict, name):
    return
roberta_model.RobertaModel.upgrade_state_dict_named = _no_upgrade


ckpt_path = "log_dir/nodp/checkpoint_best.pt"

models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0].eval()

print("model type:", type(model))
print("forward signature:", inspect.signature(model.forward))

# 打印 classification head 相关模块
for name, module in model.named_modules():
    if "head" in name.lower() or "class" in name.lower():
        print(name, type(module))

# 如果存在 classification_heads 字典，也打印一下 key
if hasattr(model, "classification_heads"):
    print("classification_heads keys:", list(model.classification_heads.keys()))
