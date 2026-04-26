import os
import sys
import time
import glob
import multiprocessing as mp
import multiprocessing.connection
import ctypes
import numpy as np
import threading
import concurrent.futures
from collections import deque
import pickle 

# 🚀 OS MEMORY FIX: Force thread stacks to 256KB to prevent Thread Stack Fragmentation
threading.stack_size(256 * 1024)

from helper import BOARD_SIZE, SHAPES

# Force Triton to use the DGX's native Blackwell-compatible assembler
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"

# config
BLOCKS = 8
FILTERS = 64

MAX_ORDER = 7
NUM_WORKERS = 20

MAX_CAPACITY = 2048

NUM_THREADS = 1
TOTAL_THREADS = NUM_WORKERS * NUM_THREADS

# ==========================================
# 1. Game Generation & Workers
# ==========================================
def generate_expert_game(bot):
    states, players = [], []
    board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    inventories = {i: list(SHAPES.keys()) for i in range(1, 5)} 
    first_moves = {1: True, 2: True, 3: True, 4: True}
    current_player = 1
    pass_count = 0
    
    while pass_count < 4:
        legal_moves = bot._get_legal_moves(board, current_player, inventories, first_moves)
        if not legal_moves:
            pass_count += 1
            current_player = (current_player % 4) + 1
            continue
            
        pass_count = 0
        is_fast_playout = np.random.rand() < 0.80 
        
        state_tensor = bot._build_state_tensor(board, current_player, inventories, first_moves)
        action = bot.get_action(board, current_player, inventories, first_moves, legal_moves, is_training=True, fast_playout=is_fast_playout)
        
        if not is_fast_playout:
            states.append(state_tensor)
            players.append(current_player)
            
        shape_name, coords = bot._decode_action(action, legal_moves)
        for r, c in coords: board[r][c] = current_player
        if shape_name in inventories[current_player]: inventories[current_player].remove(shape_name)
        first_moves[current_player] = False
        current_player = (current_player % 4) + 1
        
    def get_score(pid): return sum(int(s.split('_')[0]) for s in inventories.get(pid, []))
    team1_score, team2_score = get_score(1) + get_score(3), get_score(2) + get_score(4)
    
    val_targets, score_targets = [], []
    for p in players:
        if p in [1, 3]:
            val_targets.append(1 if team1_score < team2_score else -1)
            score_targets.append(team2_score - team1_score) 
        else:
            val_targets.append(1 if team2_score < team1_score else -1)
            score_targets.append(team1_score - team2_score)

    total_turns_played = 84 - sum(len(inv) for inv in inventories.values())
    return states, val_targets, score_targets, total_turns_played

def single_game_thread(games_per_thread, conn, thread_idx, shared_data_bases, total_threads, shared_counter):
    from tf_alphazero_bot import ExpertBlokusBot
    
    shared_states = np.ctypeslib.as_array(shared_data_bases[0]).reshape((total_threads, MAX_CAPACITY, 20, 20, 8))
    shared_values = np.ctypeslib.as_array(shared_data_bases[1]).reshape((total_threads, MAX_CAPACITY))
    shared_scores = np.ctypeslib.as_array(shared_data_bases[2]).reshape((total_threads, MAX_CAPACITY))
    
    bot = ExpertBlokusBot(pipe=conn, shared_data=(thread_idx, shared_states, shared_values, shared_scores), is_training=True)
    
    S, V, SC = [], [], []
    thread_total_turns = 0 
    
    for _ in range(games_per_thread):
        s, v, sc, turns = generate_expert_game(bot)
        S.extend(s); V.extend(v); SC.extend(sc)
        thread_total_turns += turns
        with shared_counter.get_lock(): shared_counter.value += 1
            
    conn.send("DONE")
    
    if hasattr(bot, 'executor') and bot.executor is not None:
        bot.executor.shutdown(wait=True)
    return S, V, SC, thread_total_turns

def distributed_train_worker(conns, result_queue, thread_indices, shared_counter, shared_data_bases, total_threads, cmd_queue):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    import threading
    import concurrent.futures
    threading.stack_size(256 * 1024)
    
    while True:
        cmd = cmd_queue.get()
        if cmd == "STOP":
            break
            
        games_per_thread = cmd
        S, V, SC = [], [], []
        worker_total_turns = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(conns)) as executor:
            futures = []
            for conn, tid in zip(conns, thread_indices):
                futures.append(executor.submit(single_game_thread, games_per_thread, conn, tid, shared_data_bases, total_threads, shared_counter))
            
            for f in concurrent.futures.as_completed(futures):
                s, v, sc, turns = f.result()
                S.extend(s); V.extend(v); SC.extend(sc)
                worker_total_turns += turns
                
        result_queue.put((S, V, SC, worker_total_turns))

# ==========================================
# 2. Inference Server
# ==========================================
def training_inference_server(conns, model, device, shared_counter, total_games, shared_data_bases, total_threads):
    import torch 
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    shared_states = np.ctypeslib.as_array(shared_data_bases[0]).reshape((total_threads, MAX_CAPACITY, 20, 20, 8))
    shared_values = np.ctypeslib.as_array(shared_data_bases[1]).reshape((total_threads, MAX_CAPACITY))
    shared_scores = np.ctypeslib.as_array(shared_data_bases[2]).reshape((total_threads, MAX_CAPACITY))

    active_conns = list(conns)
    conn_to_id = {conns[i]: i for i in range(total_threads)}
    
    ready_indices, batch_sizes, ready_pipes = [], [], []
    total_states_queued = 0
    last_print_time = time.time()

    CHUNK_SIZE =  2 ** MAX_ORDER
    MIN_BATCH_SIZE = 64
    is_cuda = device.type == 'cuda'

    # 🚀 SETUP: Pre-allocate static GPU buffer for zero-overhead padding required by CUDA Graphs
    if is_cuda:
        static_input_buffer = torch.zeros((CHUNK_SIZE, 20, 20, 8), dtype=torch.float32, device=device)

    while active_conns:
        readable = multiprocessing.connection.wait(active_conns, timeout=0.0001)
        for p in readable:
            try:
                msg = p.recv()
                if msg == "DONE":
                    active_conns.remove(p)
                else:
                    ready_indices.append(conn_to_id[p])
                    batch_sizes.append(msg) 
                    ready_pipes.append(p)
                    total_states_queued += msg
            except EOFError:
                if p in active_conns: active_conns.remove(p)

        if total_states_queued > 0:
            t0 = time.time()
            
            # 🚀 DGX UMA OPTIMIZATION: Shift Concatenation entirely to the GPU
            gpu_views = [
                torch.from_numpy(shared_states[w_id, :size]).to(device, non_blocking=True)
                for w_id, size in zip(ready_indices, batch_sizes)
            ]
            batch_tensor_gpu = torch.cat(gpu_views, dim=0)
            
            actual_size = batch_tensor_gpu.shape[0]
            t_copy = (time.time() - t0) * 1000 
            
            t1 = time.time()
            
            # 🚀 DGX BULK TRANSFER: Pre-allocate output tensors
            all_v = torch.empty(actual_size, dtype=torch.float32, device=device)
            all_s = torch.empty(actual_size, dtype=torch.float32, device=device)
            
            tensor_cursor = 0
            rem = actual_size
            
            while rem > 0:
                curr_size = min(rem, CHUNK_SIZE)
                
                pad_target = 1 << (curr_size - 1).bit_length() if curr_size > 0 else 0
                if pad_target < MIN_BATCH_SIZE: 
                    pad_target = MIN_BATCH_SIZE
                
                chunk_torch = batch_tensor_gpu[tensor_cursor : tensor_cursor + curr_size]
                
                if is_cuda:
                    # 🚀 ZERO-ALLOCATION STATIC PADDING
                    static_input_buffer[:curr_size].copy_(chunk_torch)
                    if pad_target > curr_size:
                        static_input_buffer[curr_size:pad_target].zero_()
                    chunk_padded = static_input_buffer[:pad_target]
                else:
                    if pad_target > curr_size:
                        pad_array = torch.zeros((pad_target - curr_size, 20, 20, 8), dtype=torch.float32, device=device)
                        chunk_padded = torch.cat([chunk_torch, pad_array], dim=0)
                    else:
                        chunk_padded = chunk_torch
                    
                with torch.no_grad():
                    if is_cuda:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            pred_v, pred_s = model(chunk_padded)
                    else:
                        pred_v, pred_s = model(chunk_padded)
                        
                all_v[tensor_cursor : tensor_cursor + curr_size] = pred_v.flatten()[:curr_size]
                all_s[tensor_cursor : tensor_cursor + curr_size] = pred_s.flatten()[:curr_size]
                
                tensor_cursor += curr_size
                rem -= curr_size
                
            if is_cuda:
                torch.cuda.synchronize()
            t_infer = (time.time() - t1) * 1000 
            t_per_sample = (t_infer / actual_size) if actual_size > 0 else 0.0
            
            t2 = time.time()
            
            v_numpy = all_v.cpu().numpy()
            sc_numpy = all_s.cpu().numpy()
            
            cursor = 0
            for w_id, size in zip(ready_indices, batch_sizes):
                shared_values[w_id, :size] = v_numpy[cursor:cursor+size]
                shared_scores[w_id, :size] = sc_numpy[cursor:cursor+size]
                cursor += size
                
            t_write = (time.time() - t2) * 1000

            for pipe in ready_pipes: pipe.send(True)

            curr_time = time.time()
            if curr_time - last_print_time > 0.5:
                games_completed = shared_counter.value
                print(f"⚡ GPU BATCH {actual_size:<5} | "
                      f"Concat: {t_copy:>5.1f}ms | "
                      f"Infer: {t_infer:>5.1f}ms ({t_per_sample:>4.4f}ms/st) | "
                      f"Write: {t_write:>5.1f}ms | "
                      f"Games: {games_completed}/{total_games}   ", end='\r', flush=True)
                last_print_time = curr_time

            ready_indices, batch_sizes, ready_pipes = [], [], []
            total_states_queued = 0
            
    print("\n✅ All GPU processes finished gathering data.", flush=True)

# ==========================================
# 3. Main Training Pipeline
# ==========================================
def run_training_pipeline(num_iteration=2000):
    ctx = mp.get_context('spawn')
    
    print("🔥 Allocating Static Inter-Process Communication Arrays (Once)...", flush=True)
    shared_counter = ctx.Value('i', 0)
    
    shared_states_base = ctx.Array(ctypes.c_float, TOTAL_THREADS * MAX_CAPACITY * 20 * 20 * 8, lock=False)
    shared_values_base = ctx.Array(ctypes.c_float, TOTAL_THREADS * MAX_CAPACITY, lock=False)
    shared_scores_base = ctx.Array(ctypes.c_float, TOTAL_THREADS * MAX_CAPACITY, lock=False)
    
    shared_data_bases = (shared_states_base, shared_values_base, shared_scores_base)

    pipes = [ctx.Pipe() for _ in range(TOTAL_THREADS)]
    parent_conns, child_conns = [p[0] for p in pipes], [p[1] for p in pipes]
    
    cmd_queue = ctx.Queue()
    result_queue = ctx.Queue()
    processes = []
    
    print("🚀 STARTING WORKER PROCESSES (BEFORE PYTORCH INITIALIZATION)...", flush=True)
    for i in range(NUM_WORKERS):
        conns_for_worker = child_conns[i * NUM_THREADS : (i + 1) * NUM_THREADS]
        indices_for_worker = list(range(i * NUM_THREADS, (i + 1) * NUM_THREADS))
        p = ctx.Process(target=distributed_train_worker, args=(conns_for_worker, result_queue, indices_for_worker, shared_counter, shared_data_bases, TOTAL_THREADS, cmd_queue))
        p.start()
        processes.append(p)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

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

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔥 Using device: {device}", flush=True)

    model = PyTorchAdvancedBlokusModel(num_blocks=BLOCKS, filters=FILTERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    model_path = "blokus_expert_latest.pt"
    if os.path.exists(model_path):
        print(f"🚀 Found existing model! Resuming training from {model_path}...", flush=True)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("🚀 No existing model found. Building a new PyTorch AdvancedBlokusModel from scratch...", flush=True)

    is_mac = sys.platform == "darwin"
    
    if not is_mac and hasattr(torch, "compile"):
        import torch._dynamo
        # Raise cache limit to accommodate the various fixed bucket sizes during generation AND training batches
        torch._dynamo.config.cache_size_limit = 64
        
        print(f"🔥 Wrapping model with torch.compile (Triton Fusion + CUDA Graphs Enabled)...", flush=True)
        
        # 🚀 FIX: Move max_autotune inside the options dictionary to prevent the RuntimeError
        fast_infer_model = torch.compile(
            model, 
            fullgraph=True,
            dynamic=False,  # Enforce strictly static shapes for CUDA Graphs to work optimally
            options={
                "max_autotune": True, 
                "triton.cudagraphs": True
            }
        )
    else:
        fast_infer_model = model

    print(f"🔥 Pre-compiling Power-of-2 Buckets (64 to {2 ** MAX_ORDER}) into System RAM...", flush=True)
    fast_infer_model.eval()
    with torch.no_grad():
        for o in range(6, MAX_ORDER + 1):
            dummy = torch.zeros((2 ** o, 20, 20, 8), dtype=torch.float32, device=device)
            if device.type == 'cuda':
                # 🚀 MUST INCLUDE AUTOCAST IN WARMUP SO THE GRAPH CACHES THE FP16 PATH
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _ = fast_infer_model(dummy)
            else:
                _ = fast_infer_model(dummy)
    print("✅ All dynamic graphs successfully compiled and cached!", flush=True)
    
    TOTAL_GAMES_PER_ITERATION = 500
    games_per_thread = max(1, TOTAL_GAMES_PER_ITERATION // TOTAL_THREADS)
    actual_total_games = TOTAL_THREADS * games_per_thread

    REPLAY_BUFFER_SIZE = 100000
    start_iteration = 1
    
    buffer_path = "replay_buffer.pkl"
    if os.path.exists(buffer_path):
        print(f"💾 Found existing Replay Buffer! Loading historical states from {buffer_path}...", flush=True)
        with open(buffer_path, "rb") as f:
            data = pickle.load(f)
            if len(data) == 4:
                replay_states, replay_values, replay_scores, start_iteration = data
            else:
                replay_states, replay_values, replay_scores = data
                start_iteration = 1
        print(f"✅ Successfully loaded {len(replay_states)} historical states. Resuming at Iteration {start_iteration}.", flush=True)
    else:
        print(f"💾 No existing Replay Buffer found. Initializing empty queues.", flush=True)
        replay_states = deque(maxlen=REPLAY_BUFFER_SIZE)
        replay_values = deque(maxlen=REPLAY_BUFFER_SIZE)
        replay_scores = deque(maxlen=REPLAY_BUFFER_SIZE)

    mse_loss_fn = nn.MSELoss()
    huber_loss_fn = nn.HuberLoss(delta=1.0)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    for iteration in range(start_iteration, start_iteration + num_iteration):
        
        current_lr = max(2e-4 * (0.95 ** (iteration - 1)), 1e-5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        print(f"\n" + "="*70, flush=True)
        print(f"🚀 STARTING Q-LEARNING ITERATION {iteration} | LR: {current_lr:.2e} | Generating {actual_total_games} Games", flush=True)
        print("="*70, flush=True)
        
        start_time = time.time()
        shared_counter.value = 0
        
        for _ in range(NUM_WORKERS):
            cmd_queue.put(games_per_thread)

        fast_infer_model.eval()
        training_inference_server(parent_conns, fast_infer_model, device, shared_counter, actual_total_games, shared_data_bases, TOTAL_THREADS)

        S, V, SC = [], [], []
        iteration_total_turns = 0
        
        for _ in range(NUM_WORKERS):
            s, v, sc, turns = result_queue.get()
            S.extend(s); V.extend(v); SC.extend(sc)
            iteration_total_turns += turns
        
        generation_time = time.time() - start_time
        avg_game_length = iteration_total_turns / actual_total_games
        
        print(f"\n✅ Data Gen Complete in {generation_time:.1f}s | Extracted {len(S)} New States | EXACT Avg Turns/Game: {avg_game_length:.1f}", flush=True)
        
        replay_states.extend(S)
        replay_values.extend(V)
        replay_scores.extend(SC)
        
        current_buffer_size = len(replay_states)
        print(f"🧠 Replay Buffer Size: {current_buffer_size} / {REPLAY_BUFFER_SIZE}. Constructing Pipeline...", flush=True)
        
        tensor_x = torch.from_numpy(np.array(replay_states, dtype=np.float32))
        tensor_v = torch.from_numpy(np.array(replay_values, dtype=np.float32).reshape(-1, 1))
        tensor_s = torch.from_numpy(np.array(replay_scores, dtype=np.float32).reshape(-1, 1))
        
        dataset = TensorDataset(tensor_x, tensor_v, tensor_s)
        dataloader = DataLoader(dataset, batch_size=8192, shuffle=True, pin_memory=(device.type == 'cuda'), num_workers=0)

        print(f"🧠 Training Q-Value Neural Network...", flush=True)
        # Ensure base model trains correctly without compiling overhead during backward pass
        model.train()
        
        epochs = 4
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_yv, batch_ys in dataloader:
                batch_x, batch_yv, batch_ys = batch_x.to(device), batch_yv.to(device), batch_ys.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                
                if device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        pred_v, pred_s = model(batch_x)
                        loss_v = mse_loss_fn(pred_v, batch_yv)
                        loss_s = huber_loss_fn(pred_s, batch_ys)
                        loss = loss_v + 0.05 * loss_s
                        
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred_v, pred_s = model(batch_x)
                    loss_v = mse_loss_fn(pred_v, batch_yv)
                    loss_s = huber_loss_fn(pred_s, batch_ys)
                    loss = loss_v + 0.05 * loss_s
                    loss.backward()
                    optimizer.step()
                    
                epoch_loss += loss.item()
                num_batches += 1
                
            print(f"   Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss/num_batches:.4f}", flush=True)
        
        print(f"💾 Saving generation {iteration} model and replay buffer...", flush=True)
        current_model_name = f"blokus_expert_v{iteration}.pt"
        
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration
        }
        
        torch.save(checkpoint_data, current_model_name)
        torch.save(checkpoint_data, "blokus_expert_latest.pt") 

        with open(buffer_path, "wb") as f:
            pickle.dump((replay_states, replay_values, replay_scores, iteration + 1), f, protocol=pickle.HIGHEST_PROTOCOL)

        for old_file in glob.glob("blokus_expert_v*.pt"):
            if old_file != current_model_name and old_file != "blokus_expert_latest.pt":
                try: 
                    os.remove(old_file)
                    print(f"🗑️ Auto-deleted old model: {old_file}", flush=True)
                except OSError: pass

    for _ in range(NUM_WORKERS):
        cmd_queue.put("STOP")
    for p in processes:
        p.join()

if __name__ == "__main__":
    mp.freeze_support()
    run_training_pipeline()
