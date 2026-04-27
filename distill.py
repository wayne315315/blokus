import os
import sys
import glob
import time
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# Global import for Dynamo to prevent UnboundLocalError
if hasattr(torch, "compile"):
    import torch._dynamo

# 🚀 IMPORT MODEL FROM model.py
from model import PyTorchAdvancedBlokusModel
from const import BLOCKS, FILTERS

# Force Triton to use the DGX's native Blackwell-compatible assembler
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"

# config
STUDENT_BLOCKS = 8
STUDENT_FILTERS = 64

# ==========================================
# Knowledge Distillation Pipeline
# ==========================================
def run_distillation(epochs=10, batch_size=8192):
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔥 Distilling on device: {device}", flush=True)

    # --- 1. Load the Teacher Model (Small) ---
    print(f"\n📚 Loading Teacher Model ({BLOCKS} Blocks, {FILTERS} Filters)...", flush=True)
    teacher_model = PyTorchAdvancedBlokusModel(num_blocks=BLOCKS, filters=FILTERS).to(device)
    
    teacher_path = "blokus_expert_latest.pt"
    if not os.path.exists(teacher_path):
        print(f"❌ Teacher model {teacher_path} not found. Aborting.", flush=True)
        return

    checkpoint = torch.load(teacher_path, map_location=device, weights_only=True)
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model.eval() 
    
    print(f"🎓 Initializing Student Model ({STUDENT_BLOCKS} Blocks, {STUDENT_FILTERS} Filters)...", flush=True)
    student_model = PyTorchAdvancedBlokusModel(num_blocks=STUDENT_BLOCKS, filters=STUDENT_FILTERS).to(device)
    student_model.train()
    
    optimizer = optim.Adam(student_model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # --- 3. Apply Triton Fusion & CUDA Graphs ---
    is_mac = sys.platform == "darwin"
    if not is_mac and hasattr(torch, "compile"):
        torch._dynamo.config.cache_size_limit = 32
        
        print("🔥 Compiling Teacher and Student models for Triton Fusion + CUDA Graphs...", flush=True)
        
        teacher_model = torch.compile(
            teacher_model, 
            fullgraph=True,
            dynamic=False, 
            options={"max_autotune": True, "triton.cudagraphs": True}
        )
        
        student_model = torch.compile(
            student_model, 
            fullgraph=True,
            dynamic=False, 
            options={"max_autotune": True, "triton.cudagraphs": True}
        )
        
        print("⏳ Warming up Static Graphs...", flush=True)
        dummy_batch = torch.zeros((batch_size, 20, 20, 8), dtype=torch.float32, device=device)
        
        if device.type == 'cuda':
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _ = teacher_model(dummy_batch)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _ = student_model(dummy_batch)
        else:
            with torch.no_grad():
                _ = teacher_model(dummy_batch)
            _ = student_model(dummy_batch)
            
        print("✅ Compilation and graph capture successful!", flush=True)


    # --- 4. Aggregate all local cluster Replay Buffers ---
    print("\n💾 Scanning for Replay Buffers...", flush=True)
    buffer_files = glob.glob("replay_buffer*.pkl")
    
    if not buffer_files:
        print("❌ No replay buffers found. Need historical data to distill.", flush=True)
        return
        
    all_states, all_true_v, all_true_s = [], [], []
    
    for b_file in buffer_files:
        print(f"   -> Loading {b_file}...", flush=True)
        with open(b_file, "rb") as f:
            data = pickle.load(f)
            if len(data) == 4:
                states, values, scores, _ = data
            else:
                states, values, scores = data
            
            all_states.extend(states)
            all_true_v.extend(values)
            all_true_s.extend(scores)
            
    total_states = len(all_states)
    print(f"✅ Merged {len(buffer_files)} buffers into a dataset of {total_states} states.", flush=True)

    # --- 5. Build the Dataset ---
    tensor_x = torch.from_numpy(np.array(all_states, dtype=np.float32))
    tensor_true_v = torch.from_numpy(np.array(all_true_v, dtype=np.float32).reshape(-1, 1))
    tensor_true_s = torch.from_numpy(np.array(all_true_s, dtype=np.float32).reshape(-1, 1))
    
    dataset = TensorDataset(tensor_x, tensor_true_v, tensor_true_s)
    
    # 🚀 SECRETS TO TRAINING CUDA GRAPHS: drop_last=True ensures strictly static batch shapes
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        pin_memory=(device.type == 'cuda'), 
        num_workers=2
    )

    mse_loss_fn = nn.MSELoss()
    huber_loss_fn = nn.HuberLoss(delta=1.0)

    # --- 6. The Distillation Loop ---
    print("\n🚀 Beginning Knowledge Distillation...", flush=True)
    
    ALPHA_TEACHER = 0.70 
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()
        
        for batch_x, batch_true_v, batch_true_s in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_true_v = batch_true_v.to(device, non_blocking=True)
            batch_true_s = batch_true_s.to(device, non_blocking=True)
            
            with torch.no_grad():
                if device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        teacher_v, teacher_s = teacher_model(batch_x)
                else:
                    teacher_v, teacher_s = teacher_model(batch_x)

            optimizer.zero_grad(set_to_none=True)
            
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    student_v, student_s = student_model(batch_x)
                    
                    loss_kd_v = mse_loss_fn(student_v, teacher_v)
                    loss_kd_s = mse_loss_fn(student_s, teacher_s)
                    
                    loss_true_v = mse_loss_fn(student_v, batch_true_v)
                    loss_true_s = huber_loss_fn(student_s, batch_true_s)
                    
                    loss_kd_total = loss_kd_v + 0.05 * loss_kd_s
                    loss_true_total = loss_true_v + 0.05 * loss_true_s
                    
                    loss = (ALPHA_TEACHER * loss_kd_total) + ((1.0 - ALPHA_TEACHER) * loss_true_total)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                student_v, student_s = student_model(batch_x)
                loss_kd_v = mse_loss_fn(student_v, teacher_v)
                loss_kd_s = mse_loss_fn(student_s, teacher_s)
                loss_true_v = mse_loss_fn(student_v, batch_true_v)
                loss_true_s = huber_loss_fn(student_s, batch_true_s)
                
                loss_kd_total = loss_kd_v + 0.05 * loss_kd_s
                loss_true_total = loss_true_v + 0.05 * loss_true_s
                loss = (ALPHA_TEACHER * loss_kd_total) + ((1.0 - ALPHA_TEACHER) * loss_true_total)
                
                loss.backward()
                optimizer.step()
                
            epoch_loss += loss.item()
            num_batches += 1
            
        t1 = time.time()
        print(f"   Epoch {epoch+1}/{epochs} | Distillation Loss: {epoch_loss/num_batches:.5f} | Time: {t1-t0:.2f}s", flush=True)

    # --- 7. Save the Upgraded Model ---
    student_path = "blokus_student_latest.pt"
    print(f"\n💾 Saving upgraded student model to {student_path}...", flush=True)
    
    # Retrieve the uncompiled base model to save clean weights
    clean_model = student_model._orig_mod if hasattr(student_model, '_orig_mod') else student_model
    
    checkpoint_data = {
        'model_state_dict': clean_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': 1 
    }
    torch.save(checkpoint_data, student_path)
    print("✅ Distillation Complete! Rename 'blokus_student_latest.pt' to 'blokus_expert_latest.pt' and spin up the cluster.", flush=True)

if __name__ == "__main__":
    run_distillation()
