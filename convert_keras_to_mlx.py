import tensorflow as tf
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
import numpy as np

print("Loading TensorFlow Keras model...")
tf_model = tf.keras.models.load_model("blokus_expert_latest.keras", compile=False)

# ==========================================
# 1. Extract weights purely by physical dimensions
# ==========================================
print("Extracting matrices by physical shape...")

# Conv2D and BatchNorm are strictly sequential in your architecture
conv_weights = [l.get_weights()[0] for l in tf_model.layers if type(l).__name__ == 'Conv2D']
bn_weights = [l.get_weights() for l in tf_model.layers if type(l).__name__ == 'BatchNormalization']

# Isolate Dense layers based on their mathematical shape
dense_layers = [l for l in tf_model.layers if type(l).__name__ == 'Dense']
dense_16 = [l.get_weights() for l in dense_layers if l.get_weights()[0].shape[0] == 16]
dense_400 = [l.get_weights() for l in dense_layers if l.get_weights()[0].shape[0] == 400]
dense_256 = [l.get_weights() for l in dense_layers if l.get_weights()[0].shape[0] == 256]

print(f"Extracted {len(conv_weights)} Conv2D matrices. (Expected: 11)")
print(f"Extracted {len(bn_weights)} BatchNorm sets. (Expected: 11)")
print(f"Extracted {len(dense_16)} ResBlock Dense matrices. (Expected: 4)")
print(f"Extracted {len(dense_400)} Head Hidden Dense matrices. (Expected: 2)")
print(f"Extracted {len(dense_256)} Head Output Dense matrices. (Expected: 2)")

# ==========================================
# 2. Formatting Helpers (TF -> MLX)
# ==========================================
def fmt_conv(w): 
    # TF: (H, W, in_C, out_C) -> MLX: (out_C, H, W, in_C)
    return mx.array(w.transpose(3, 0, 1, 2))

def fmt_bn(w): 
    return mx.array(w[0]), mx.array(w[1]), mx.array(w[2]), mx.array(w[3])

def fmt_dense(w, use_bias=True):
    # TF: (in, out) -> MLX: (out, in)
    mw = mx.array(w[0].transpose(1, 0))
    if use_bias: return mw, mx.array(w[1])
    return mw

# ==========================================
# 3. Map to MLX Architecture
# ==========================================
print("Mapping weights to Apple MLX Architecture...")
params = []

# Initial Block
params.append(("conv_init.weight", fmt_conv(conv_weights.pop(0))))
g, b, m, v = fmt_bn(bn_weights.pop(0))
params.extend([("bn_init.weight", g), ("bn_init.bias", b), ("bn_init.running_mean", m), ("bn_init.running_var", v)])

# Residual Blocks (4 blocks)
for i in range(4):
    params.append((f"res_blocks.{i}.conv1.weight", fmt_conv(conv_weights.pop(0))))
    g, b, m, v = fmt_bn(bn_weights.pop(0))
    params.extend([(f"res_blocks.{i}.bn1.weight", g), (f"res_blocks.{i}.bn1.bias", b), (f"res_blocks.{i}.bn1.running_mean", m), (f"res_blocks.{i}.bn1.running_var", v)])
    
    params.append((f"res_blocks.{i}.conv2.weight", fmt_conv(conv_weights.pop(0))))
    g, b, m, v = fmt_bn(bn_weights.pop(0))
    params.extend([(f"res_blocks.{i}.bn2.weight", g), (f"res_blocks.{i}.bn2.bias", b), (f"res_blocks.{i}.bn2.running_mean", m), (f"res_blocks.{i}.bn2.running_var", v)])
    
    # ResBlock Dense (16x16, no bias)
    params.append((f"res_blocks.{i}.dense_g.weight", fmt_dense(dense_16.pop(0), use_bias=False)))

# Value Head
params.append(("conv_v.weight", fmt_conv(conv_weights.pop(0))))
g, b, m, v = fmt_bn(bn_weights.pop(0))
params.extend([("bn_v.weight", g), ("bn_v.bias", b), ("bn_v.running_mean", m), ("bn_v.running_var", v)])

# Hidden Dense (400x256)
w, bias = fmt_dense(dense_400.pop(0), use_bias=True)
params.extend([("dense_v1.weight", w), ("dense_v1.bias", bias)])

# Output Dense (256x1)
w, bias = fmt_dense(dense_256.pop(0), use_bias=True)
params.extend([("dense_v2.weight", w), ("dense_v2.bias", bias)])

# Score Head
params.append(("conv_s.weight", fmt_conv(conv_weights.pop(0))))
g, b, m, v = fmt_bn(bn_weights.pop(0))
params.extend([("bn_s.weight", g), ("bn_s.bias", b), ("bn_s.running_mean", m), ("bn_s.running_var", v)])

# Hidden Dense (400x256)
w, bias = fmt_dense(dense_400.pop(0), use_bias=True)
params.extend([("dense_s1.weight", w), ("dense_s1.bias", bias)])

# Output Dense (256x1)
w, bias = fmt_dense(dense_256.pop(0), use_bias=True)
params.extend([("dense_s2.weight", w), ("dense_s2.bias", bias)])

# ==========================================
# 4. Define MLX Model and Save
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(filters)
        self.dense_g = nn.Linear(filters, filters, bias=False)

    def __call__(self, x):
        shortcut = x
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        g = x.mean(axis=(1, 2))
        g = self.dense_g(g).reshape(g.shape[0], 1, 1, g.shape[1])
        return nn.relu(x + g + shortcut)

class MLXBlokusModel(nn.Module):
    def __init__(self, filters=16, num_blocks=4):
        super().__init__()
        self.conv_init = nn.Conv2d(8, filters, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm(filters)
        self.res_blocks = [ResidualBlock(filters) for _ in range(num_blocks)]
        self.conv_v = nn.Conv2d(filters, 1, 1, padding=0, bias=False)
        self.bn_v = nn.BatchNorm(1)
        self.dense_v1 = nn.Linear(400, 256)
        self.dense_v2 = nn.Linear(256, 1)
        self.conv_s = nn.Conv2d(filters, 1, 1, padding=0, bias=False)
        self.bn_s = nn.BatchNorm(1)
        self.dense_s1 = nn.Linear(400, 256)
        self.dense_s2 = nn.Linear(256, 1)

    def __call__(self, x):
        x = nn.relu(self.bn_init(self.conv_init(x)))
        for block in self.res_blocks: x = block(x)
        v = nn.relu(self.bn_v(self.conv_v(x)))
        v = v.reshape(v.shape[0], -1)
        v = nn.relu(self.dense_v1(v))
        value_out = mx.tanh(self.dense_v2(v))
        s = nn.relu(self.bn_s(self.conv_s(x)))
        s = s.reshape(s.shape[0], -1)
        s = nn.relu(self.dense_s1(s))
        score_out = self.dense_s2(s)
        return value_out, score_out

mlx_model = MLXBlokusModel()
mlx_model.update(tree_unflatten(params))

# Force evaluation to guarantee arrays are allocated in Unified Memory
mx.eval(mlx_model.parameters())

mlx_model.save_weights("blokus_expert_latest.safetensors")
print("✅ Conversion successfully completed! Saved as blokus_expert_latest.safetensors")