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

# 🍏 Import Apple MLX framework
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

threading.stack_size(256 * 1024)

from helper import BOARD_SIZE, SHAPES

MAX_ORDER = 10
_NUM_CPUS = mp.cpu_count()
NUM_WORKERS = min(31, _NUM_CPUS - 1 if _NUM_CPUS > 1 else 1)
MAX_CAPACITY = 2048
NUM_THREADS = 4
TOTAL_THREADS = NUM_WORKERS * NUM_THREADS

# ==========================================
# 1. MLX Neural Network Architecture
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
        x = x + g + shortcut
        return nn.relu(x)

class MLXAdvancedBlokusModel(nn.Module):
    def __init__(self, board_size=20, num_blocks=4, filters=16):
        super().__init__()
        self.conv_init = nn.Conv2d(8, filters, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm(filters)
        self.res_blocks = [ResidualBlock(filters) for _ in range(num_blocks)]
        
        self.conv_v = nn.Conv2d(filters, 1, 1, padding=0, bias=False)
        self.bn_v = nn.BatchNorm(1)
        self.dense_v1 = nn.Linear(board_size * board_size * 1, 256)
        self.dense_v2 = nn.Linear(256, 1)

        self.conv_s = nn.Conv2d(filters, 1, 1, padding=0, bias=False)
        self.bn_s = nn.BatchNorm(1)
        self.dense_s1 = nn.Linear(board_size * board_size * 1, 256)
        self.dense_s2 = nn.Linear(256, 1)

    def __call__(self, x):
        x = nn.relu(self.bn_init(self.conv_init(x)))
        for block in self.res_blocks:
            x = block(x)

        v = nn.relu(self.bn_v(self.conv_v(x)))
        v = v.reshape(v.shape[0], -1)
        v = nn.relu(self.dense_v1(v))
        value_out = mx.tanh(self.dense_v2(v))

        s = nn.relu(self.bn_s(self.conv_s(x)))
        s = s.reshape(s.shape[0], -1)
        s = nn.relu(self.dense_s1(s))
        score_out = self.dense_s2(s)

        return value_out, score_out

# ==========================================
# 2. Loss Functions (No global compilation)
# ==========================================
def huber_loss(predictions, targets, delta=1.0):
    error = predictions - targets
    abs_error = mx.abs(error)
    quadratic = mx.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return mx.mean(0.5 * mx.square(quadratic) + delta * linear)

def loss_fn(model, x, y_val, y_score):
    pred_val, pred_score = model(x)
    loss_v = mx.mean(mx.square(pred_val - y_val))
    loss_s = huber_loss(pred_score, y_score)
    return loss_v + 0.05 * loss_s

# ==========================================
# 3. Game Generation & Workers
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
    shared_states = np.ctypeslib.as_array(shared_data_bases[0].get_obj()).reshape((total_threads, MAX_CAPACITY, 20, 20, 8))
    shared_values = np.ctypeslib.as_array(shared_data_bases[1].get_obj()).reshape((total_threads, MAX_CAPACITY))
    shared_scores = np.ctypeslib.as_array(shared_data_bases[2].get_obj()).reshape((total_threads, MAX_CAPACITY))
    
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
        bot.executor.shutdown(wait=False)
    return S, V, SC, thread_total_turns

def distributed_train_worker(games_per_thread, conns, result_queue, thread_indices, shared_counter, shared_data_bases, total_threads):
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
# 4. Inference Server
# ==========================================
def training_inference_server(conns, fast_infer, shared_counter, total_games, shared_data_bases, total_threads):
    shared_states = np.ctypeslib.as_array(shared_data_bases[0].get_obj()).reshape((total_threads, MAX_CAPACITY, 20, 20, 8))
    shared_values = np.ctypeslib.as_array(shared_data_bases[1].get_obj()).reshape((total_threads, MAX_CAPACITY))
    shared_scores = np.ctypeslib.as_array(shared_data_bases[2].get_obj()).reshape((total_threads, MAX_CAPACITY))

    active_conns = list(conns)
    conn_to_id = {conns[i]: i for i in range(total_threads)}
    ready_indices, batch_sizes, ready_pipes = [], [], []
    total_states_queued = 0
    last_print_time = time.time()
    
    CHUNK_SIZE =  2 ** MAX_ORDER
    MIN_BATCH_SIZE = 64

    MAX_POSSIBLE_BATCH = total_threads * MAX_CAPACITY
    SERVER_BUFFER = np.empty((MAX_POSSIBLE_BATCH, 20, 20, 8), dtype=np.float32)
    SERVER_BUFFER.fill(0.0) 

    while active_conns:
        readable = multiprocessing.connection.wait(active_conns, timeout=0.001)
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
            actual_size = 0
            for w_id, size in zip(ready_indices, batch_sizes):
                SERVER_BUFFER[actual_size : actual_size + size] = shared_states[w_id, :size]
                actual_size += size
                
            batch_tensor = SERVER_BUFFER[:actual_size]
            t_copy = (time.time() - t0) * 1000 
            
            t1 = time.time()
            v_numpy = np.empty(actual_size, dtype=np.float32)
            sc_numpy = np.empty(actual_size, dtype=np.float32)
            tensor_cursor = 0
            
            while tensor_cursor < actual_size:
                curr_size = min(actual_size - tensor_cursor, CHUNK_SIZE)
                
                # 🍏 16-BIT PRECISION INJECTION
                if curr_size == CHUNK_SIZE:
                    chunk_input = mx.array(batch_tensor[tensor_cursor : tensor_cursor + curr_size], dtype=mx.float16)
                    pred_v, pred_s = fast_infer(chunk_input)
                else:
                    pad_target = 1 << (curr_size - 1).bit_length() if curr_size > 0 else 0
                    if pad_target < MIN_BATCH_SIZE: pad_target = MIN_BATCH_SIZE
                    
                    chunk = batch_tensor[tensor_cursor : tensor_cursor + curr_size]
                    if pad_target > curr_size:
                        pad_array = np.zeros((pad_target - curr_size, 20, 20, 8), dtype=np.float32)
                        chunk_padded = np.concatenate([chunk, pad_array], axis=0)
                        chunk_input = mx.array(chunk_padded, dtype=mx.float16)
                        pred_v, pred_s = fast_infer(chunk_input)
                    else:
                        chunk_input = mx.array(chunk, dtype=mx.float16)
                        pred_v, pred_s = fast_infer(chunk_input)
                
                mx.eval(pred_v, pred_s)
                # Results come back as float16 numpy arrays, which implicitly upcast back to our float32 buffer
                v_numpy[tensor_cursor : tensor_cursor + curr_size] = np.array(pred_v).flatten()[:curr_size]
                sc_numpy[tensor_cursor : tensor_cursor + curr_size] = np.array(pred_s).flatten()[:curr_size]
                tensor_cursor += curr_size
                
            t_infer = (time.time() - t1) * 1000 
            t_per_sample = (t_infer / actual_size) if actual_size > 0 else 0.0
            
            t2 = time.time()
            cursor = 0
            for w_id, size in zip(ready_indices, batch_sizes):
                shared_values[w_id, :size] = v_numpy[cursor:cursor+size]
                shared_scores[w_id, :size] = sc_numpy[cursor:cursor+size]
                cursor += size
            t_write = (time.time() - t2) * 1000

            for pipe in ready_pipes: pipe.send(True)

            curr_time = time.time()
            if curr_time - last_print_time > 0.5:
                print(f"⚡ GPU BATCH {actual_size:<5} | "
                      f"Copy: {t_copy:>5.1f}ms | "
                      f"Infer: {t_infer:>5.1f}ms ({t_per_sample:>4.4f}ms/st) | "
                      f"Write: {t_write:>5.1f}ms | "
                      f"Games: {shared_counter.value}/{total_games}   ", end='\r', flush=True)
                last_print_time = curr_time

            ready_indices, batch_sizes, ready_pipes = [], [], []
            total_states_queued = 0
            
    print("\n✅ All GPU processes finished gathering data.", flush=True)

# ==========================================
# 5. Main Training Pipeline
# ==========================================
def run_training_pipeline(num_iteration=1000):
    model = MLXAdvancedBlokusModel()
    
    weights_path = "blokus_expert_latest.safetensors"
    optim_path = "blokus_expert_optim.safetensors"

    if os.path.exists(weights_path):
        print(f"🚀 Found existing model! Resuming training from {weights_path}...", flush=True)
        model.load_weights(weights_path)
    
    # 🍏 FORCE FLOAT16 PRECISION
    # We cast after loading to ensure all weights reside in memory as 16-bit floats
    model.set_dtype(mx.float16)

    optimizer = optim.Adam(learning_rate=2e-4)
    if os.path.exists(optim_path):
        optimizer.load_state(optim_path)

    # 🍏 Compile closures correctly
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    @mx.compile
    def train_step(x, y_val, y_score):
        loss, grads = loss_and_grad_fn(model, x, y_val, y_score)
        optimizer.update(model, grads)
        return loss

    @mx.compile
    def fast_infer(batch_tensor): 
        return model(batch_tensor)

    print(f"🔥 Pre-compiling MLX Power-of-2 Buckets (64 to {2 ** MAX_ORDER}) into System RAM...", flush=True)
    for o in range(6, MAX_ORDER + 1):
        # 🍏 Dummy tensor must match the float16 precision
        dummy = mx.zeros((2 ** o, 20, 20, 8), dtype=mx.float16)
        v, s = fast_infer(dummy)
        mx.eval(v, s)
    print("✅ All dynamic graphs successfully compiled via @mx.compile!", flush=True)
    
    TOTAL_GAMES_PER_ITERATION = 1024
    games_per_thread = max(1, TOTAL_GAMES_PER_ITERATION // TOTAL_THREADS)
    actual_total_games = TOTAL_THREADS * games_per_thread

    REPLAY_BUFFER_SIZE = 200000
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
        print(f"✅ Successfully loaded {len(replay_states)} historical states.", flush=True)
    else:
        replay_states = deque(maxlen=REPLAY_BUFFER_SIZE)
        replay_values = deque(maxlen=REPLAY_BUFFER_SIZE)
        replay_scores = deque(maxlen=REPLAY_BUFFER_SIZE)

    ctx = mp.get_context('spawn')
    shared_counter = ctx.Value('i', 0)
    
    shared_states_base = mp.Array(ctypes.c_float, TOTAL_THREADS * MAX_CAPACITY * 20 * 20 * 8)
    shared_values_base = mp.Array(ctypes.c_float, TOTAL_THREADS * MAX_CAPACITY)
    shared_scores_base = mp.Array(ctypes.c_float, TOTAL_THREADS * MAX_CAPACITY)
    shared_data_bases = (shared_states_base, shared_values_base, shared_scores_base)

    pipes = [ctx.Pipe() for _ in range(TOTAL_THREADS)]
    parent_conns, child_conns = [p[0] for p in pipes], [p[1] for p in pipes]

    for iteration in range(start_iteration, start_iteration + num_iteration):
        
        current_lr = max(2e-4 * (0.95 ** (iteration - 1)), 1e-5)
        optimizer.learning_rate = current_lr

        print(f"\n" + "="*70, flush=True)
        print(f"🚀 STARTING MLX Q-LEARNING ITERATION {iteration} | LR: {current_lr:.2e} | Generating {actual_total_games} Games", flush=True)
        print("="*70, flush=True)
        
        start_time = time.time()
        shared_counter.value = 0
        
        result_queue = ctx.Queue()
        processes = []
        
        for i in range(NUM_WORKERS):
            conns_for_worker = child_conns[i * NUM_THREADS : (i + 1) * NUM_THREADS]
            indices_for_worker = list(range(i * NUM_THREADS, (i + 1) * NUM_THREADS))
            p = ctx.Process(target=distributed_train_worker, args=(games_per_thread, conns_for_worker, result_queue, indices_for_worker, shared_counter, shared_data_bases, TOTAL_THREADS))
            p.start()
            processes.append(p)

        training_inference_server(parent_conns, fast_infer, shared_counter, actual_total_games, shared_data_bases, TOTAL_THREADS)

        S, V, SC = [], [], []
        iteration_total_turns = 0
        
        for _ in range(NUM_WORKERS):
            s, v, sc, turns = result_queue.get()
            S.extend(s); V.extend(v); SC.extend(sc)
            iteration_total_turns += turns

        for p in processes: p.join()
        
        generation_time = time.time() - start_time
        avg_game_length = iteration_total_turns / actual_total_games
        print(f"\n✅ Data Gen Complete in {generation_time:.1f}s | Extracted {len(S)} New States | EXACT Avg Turns/Game: {avg_game_length:.1f}", flush=True)
        
        replay_states.extend(S)
        replay_values.extend(V)
        replay_scores.extend(SC)
        
        current_buffer_size = len(replay_states)
        print(f"🧠 Replay Buffer Size: {current_buffer_size} / {REPLAY_BUFFER_SIZE}. Training on MLX...", flush=True)

        batch_size = 64
        epochs = 4
        
        np_states = np.array(replay_states, dtype=np.float32)
        np_values = np.array(replay_values, dtype=np.float32).reshape(-1, 1)
        np_scores = np.array(replay_scores, dtype=np.float32).reshape(-1, 1)
        
        indices = np.arange(current_buffer_size)
        model.train()
        
        for epoch in range(epochs):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, current_buffer_size, batch_size):
                batch_idx = indices[i:i+batch_size]
                
                # 🍏 explicitly cast training inputs to float16
                batch_x = mx.array(np_states[batch_idx], dtype=mx.float16)
                batch_yv = mx.array(np_values[batch_idx], dtype=mx.float16)
                batch_ys = mx.array(np_scores[batch_idx], dtype=mx.float16)
                
                loss = train_step(batch_x, batch_yv, batch_ys)
                mx.eval(loss, model.parameters(), optimizer.state)
                epoch_loss += loss.item()
                num_batches += 1
                
            print(f"   Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss/num_batches:.4f}", flush=True)
        
        model.eval()
        
        print(f"💾 Saving generation {iteration} model and replay buffer...", flush=True)
        current_model_name = f"blokus_expert_v{iteration}.safetensors"
        model.save_weights(current_model_name)
        model.save_weights(weights_path)
        optimizer.save_state(optim_path)

        with open(buffer_path, "wb") as f:
            pickle.dump((replay_states, replay_values, replay_scores, iteration + 1), f, protocol=pickle.HIGHEST_PROTOCOL)

        for old_file in glob.glob("blokus_expert_v*.safetensors"):
            if old_file != current_model_name and old_file != weights_path and old_file != optim_path:
                try: os.remove(old_file); print(f"🗑️ Auto-deleted old model: {old_file}", flush=True)
                except OSError: pass

if __name__ == "__main__":
    mp.freeze_support()
    run_training_pipeline()