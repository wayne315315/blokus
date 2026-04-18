import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

# ==========================================
# 1. EMBEDDED PYTORCH MODEL DEFINITION
# (Matches blokus-torch/train_alphazero.py exactly)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.dense_g = nn.Linear(filters, filters, bias=False)

    def forward(self, x):
        shortcut = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        g = out.mean(dim=(2, 3))
        g = self.dense_g(g)
        g = g.view(g.size(0), g.size(1), 1, 1)
        
        out = out + g
        out = out + shortcut
        return F.relu(out)

class PyTorchAdvancedBlokusModel(nn.Module):
    def __init__(self, board_size=20, num_blocks=4, filters=16):
        super().__init__()
        self.board_size = board_size
        
        self.conv_init = nn.Conv2d(8, filters, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(filters)
        
        self.res_blocks = nn.ModuleList([ResBlock(filters) for _ in range(num_blocks)])
        
        self.conv_v = nn.Conv2d(filters, 1, 1, padding=0, bias=False)
        self.bn_v = nn.BatchNorm2d(1)
        self.dense_v1 = nn.Linear(board_size * board_size, 256)
        self.dense_v2 = nn.Linear(256, 1)
        
        self.conv_s = nn.Conv2d(filters, 1, 1, padding=0, bias=False)
        self.bn_s = nn.BatchNorm2d(1)
        self.dense_s1 = nn.Linear(board_size * board_size, 256)
        self.dense_s2 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.relu(self.bn_init(self.conv_init(x)))
        for block in self.res_blocks: x = block(x)
            
        v = F.relu(self.bn_v(self.conv_v(x)))
        v = v.view(v.size(0), -1) 
        v = F.relu(self.dense_v1(v))
        value_out = torch.tanh(self.dense_v2(v))
        
        s = F.relu(self.bn_s(self.conv_s(x)))
        s = s.view(s.size(0), -1) 
        s = F.relu(self.dense_s1(s))
        score_out = self.dense_s2(s)
        
        return value_out, score_out

# ==========================================
# 2. CONVERSION LOGIC
# ==========================================
def convert_keras_to_pytorch(keras_path="blokus_expert_latest.keras", pt_path="blokus_expert_latest.pt"):
    print(f"Loading Keras model from {keras_path}...")
    tf_model = tf.keras.models.load_model(keras_path, compile=False)
    
    # Extract layers by type to ensure perfect sequential matching
    keras_convs = [l for l in tf_model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    keras_bns = [l for l in tf_model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    keras_denses = [l for l in tf_model.layers if isinstance(l, tf.keras.layers.Dense)]

    print(f"Found Keras Layers: {len(keras_convs)} Conv2D, {len(keras_bns)} BatchNorm, {len(keras_denses)} Dense.")

    print("Initializing PyTorch model...")
    pt_model = PyTorchAdvancedBlokusModel(num_blocks=4, filters=16)

    # Flatten PyTorch components to align with Keras lists
    pt_convs = [pt_model.conv_init]
    for b in pt_model.res_blocks:
        pt_convs.extend([b.conv1, b.conv2])
    pt_convs.extend([pt_model.conv_v, pt_model.conv_s])

    pt_bns = [pt_model.bn_init]
    for b in pt_model.res_blocks:
        pt_bns.extend([b.bn1, b.bn2])
    pt_bns.extend([pt_model.bn_v, pt_model.bn_s])

    pt_denses = []
    for b in pt_model.res_blocks:
        pt_denses.append(b.dense_g)
        
    # 🚀 FIX: Correct Breadth-First mapping to match Keras topology perfectly!
    pt_denses.extend([pt_model.dense_v1, pt_model.dense_s1, pt_model.dense_v2, pt_model.dense_s2])

    print(f"Mapped PyTorch Layers: {len(pt_convs)} Conv2d, {len(pt_bns)} BatchNorm2d, {len(pt_denses)} Linear.")

    # ---------------------------------------------
    # Perform the transpositions and assignments
    # ---------------------------------------------
    print("Transferring Conv2D weights...")
    for k_layer, pt_layer in zip(keras_convs, pt_convs):
        w = k_layer.get_weights()[0]
        # Keras Conv2D: [Height, Width, In_Channels, Out_Channels]
        # PyTorch Conv2D: [Out_Channels, In_Channels, Height, Width]
        w = np.transpose(w, (3, 2, 0, 1))
        pt_layer.weight.data = torch.from_numpy(w)

    print("Transferring BatchNorm weights...")
    for k_layer, pt_layer in zip(keras_bns, pt_bns):
        w = k_layer.get_weights()
        pt_layer.weight.data = torch.from_numpy(w[0])       # gamma
        pt_layer.bias.data = torch.from_numpy(w[1])         # beta
        pt_layer.running_mean.data = torch.from_numpy(w[2]) # moving_mean
        pt_layer.running_var.data = torch.from_numpy(w[3])  # moving_var

    print("Transferring Dense weights...")
    for k_layer, pt_layer in zip(keras_denses, pt_denses):
        w = k_layer.get_weights()
        # Keras Dense: [In_Features, Out_Features]
        # PyTorch Linear: [Out_Features, In_Features]
        weight_matrix = np.transpose(w[0], (1, 0))
        pt_layer.weight.data = torch.from_numpy(weight_matrix)
        
        # If Keras layer uses bias, transfer it.
        if len(w) > 1: 
            pt_layer.bias.data = torch.from_numpy(w[1])

    print(f"Saving converted PyTorch model to {pt_path}...")
    checkpoint_data = {
        'model_state_dict': pt_model.state_dict(),
        'iteration': 1 
    }
    torch.save(checkpoint_data, pt_path)
    print("✅ Conversion Complete! You can now run train_alphazero.py or test_alphazero.py!")

if __name__ == "__main__":
    convert_keras_to_pytorch()