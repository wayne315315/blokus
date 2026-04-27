import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

# 🚀 IMPORT MODEL FROM model.py
from model import PyTorchAdvancedBlokusModel
from const import BLOCKS, FILTERS

# ==========================================
# CONVERSION LOGIC
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
    pt_model = PyTorchAdvancedBlokusModel(num_blocks=BLOCKS, filters=FILTERS)

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
