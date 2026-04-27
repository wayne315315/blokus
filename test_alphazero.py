import os
import sys
import time
import multiprocessing as mp
import multiprocessing.connection 
import ctypes
import numpy as np
import concurrent.futures

from helper import BOARD_SIZE, SHAPES
from player import BotPlayer  
from const import BLOCKS, FILTERS


MAX_ORDER = 8
NUM_WORKERS = 20

MAX_CAPACITY = 2048

NUM_THREADS = 1
TOTAL_THREADS = NUM_WORKERS * NUM_THREADS

def play_test_game(adv_bot, std_bot_1, std_bot_2, play_as_first):
    board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    inventories = { 1: list(SHAPES.keys()), 2: list(SHAPES.keys()), 3: list(SHAPES.keys()), 4: list(SHAPES.keys()) }
    first_moves = {1: True, 2: True, 3: True, 4: True}
    
    if play_as_first: color_map = {1: adv_bot, 2: std_bot_1, 3: adv_bot, 4: std_bot_2}
    else: color_map = {1: std_bot_1, 2: adv_bot, 3: std_bot_2, 4: adv_bot}
        
    current_color = 1
    pass_count = 0

    while pass_count < 4:
        active_bot = color_map[current_color]
        
        if active_bot.__class__.__name__ == 'ExpertBlokusBot':
            shape_name, coords = active_bot.get_play(board, current_color, inventories, first_moves, pass_count)
        else:
            inv_key = str(current_color) if str(current_color) in inventories else current_color
            shape_name, coords = active_bot.get_play(board, current_color, inventories[inv_key], first_moves[current_color])
        
        if shape_name is None: pass_count += 1
        else:
            for r, c in coords: board[r][c] = current_color
            inventories[current_color].remove(shape_name)
            first_moves[current_color] = False
            pass_count = 0
            
        current_color = (current_color % 4) + 1

    def get_score(col_id): return sum(-int(shape.split('_')[0]) for shape in inventories[col_id])

    if play_as_first:
        az_score = get_score(1) + get_score(3)
        std_score = get_score(2) + get_score(4)
    else:
        std_score = get_score(1) + get_score(3)
        az_score = get_score(2) + get_score(4)

    az_score_diff = (az_score - std_score)
    aw, sw, t = 0, 0, 0
    if az_score > std_score: aw = 1
    elif std_score > az_score: sw = 1
    else: t = 1
        
    return aw, sw, t, az_score_diff

def single_thread_task(games_per_thread, conn, thread_idx, shared_data_bases, total_threads, shared_counter):
    from tf_alphazero_bot import ExpertBlokusBot
    shared_states = np.ctypeslib.as_array(shared_data_bases[0]).reshape((total_threads, MAX_CAPACITY, 20, 20, 8))
    shared_values = np.ctypeslib.as_array(shared_data_bases[1]).reshape((total_threads, MAX_CAPACITY))
    shared_scores = np.ctypeslib.as_array(shared_data_bases[2]).reshape((total_threads, MAX_CAPACITY))
    
    bot = ExpertBlokusBot(pipe=conn, shared_data=(thread_idx, shared_states, shared_values, shared_scores), is_training=False)
    std_bot_1 = BotPlayer() 
    std_bot_2 = BotPlayer()

    az_wins, std_wins, ties, az_score_diff = 0, 0, 0, 0
    for i in range(games_per_thread):
        play_as_first = (i % 2 == 0)
        aw, sw, t, diff = play_test_game(bot, std_bot_1, std_bot_2, play_as_first)
        az_wins += aw; std_wins += sw; ties += t; az_score_diff += diff
        with shared_counter.get_lock(): shared_counter.value += 1

    conn.send("DONE") 
    if hasattr(bot, 'executor') and bot.executor is not None:
        bot.executor.shutdown(wait=True)
    return az_wins, std_wins, ties, az_score_diff

def distributed_test_worker(games_per_thread, worker_conns, result_queue, worker_thread_ids, shared_counter, shared_data_bases, total_threads):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    import threading
    import concurrent.futures
    threading.stack_size(256 * 1024)
    
    az_wins, std_wins, ties, az_score_diff = 0, 0, 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_conns)) as executor:
        futures = []
        for conn, tid in zip(worker_conns, worker_thread_ids):
            futures.append(executor.submit(single_thread_task, games_per_thread, conn, tid, shared_data_bases, total_threads, shared_counter))
            
        for f in concurrent.futures.as_completed(futures):
            aw, sw, t, diff = f.result()
            az_wins += aw; std_wins += sw; ties += t; az_score_diff += diff
            
    result_queue.put([az_wins, std_wins, ties, az_score_diff])

def test_inference_server(conns, model, device, shared_counter, total_games, shared_data_bases, total_threads):
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
    
    CHUNK_SIZE = 2 ** MAX_ORDER
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
            
            # Single bulk transfer back to CPU space
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
                games_remaining = total_games - shared_counter.value
                print(f"⚡ GPU INFERENCE | Batch: {actual_size:<5} | "
                      f"Concat: {t_copy:>5.1f}ms | "
                      f"Infer: {t_infer:>5.1f}ms ({t_per_sample:>4.4f}ms/st) | "
                      f"Write: {t_write:>5.1f}ms | "
                      f"Remaining: {games_remaining:<4} ", end='\r', flush=True)
                last_print_time = curr_time

            ready_indices, batch_sizes, ready_pipes = [], [], []
            total_states_queued = 0
            
    print("\n✅ All test games completed!", flush=True)

def test_model(num_games=100):
    ctx = mp.get_context('spawn')
    
    shared_counter = ctx.Value('i', 0)
    
    shared_states_base = ctx.Array(ctypes.c_float, TOTAL_THREADS * MAX_CAPACITY * 20 * 20 * 8, lock=False)
    shared_values_base = ctx.Array(ctypes.c_float, TOTAL_THREADS * MAX_CAPACITY, lock=False)
    shared_scores_base = ctx.Array(ctypes.c_float, TOTAL_THREADS * MAX_CAPACITY, lock=False)
    
    shared_data_bases = (shared_states_base, shared_values_base, shared_scores_base)

    pipes = [ctx.Pipe() for _ in range(TOTAL_THREADS)]
    parent_conns, child_conns = [p[0] for p in pipes], [p[1] for p in pipes]
    
    result_queue = ctx.Queue()
    processes = []
    games_per_thread = max(1, num_games // TOTAL_THREADS)
    
    print("🚀 STARTING WORKER PROCESSES (BEFORE PYTORCH INITIALIZATION)...", flush=True)
    for i in range(NUM_WORKERS):
        conns_for_worker = child_conns[i * NUM_THREADS : (i + 1) * NUM_THREADS]
        indices_for_worker = list(range(i * NUM_THREADS, (i + 1) * NUM_THREADS))
        p = ctx.Process(target=distributed_test_worker, args=(games_per_thread, conns_for_worker, result_queue, indices_for_worker, shared_counter, shared_data_bases, TOTAL_THREADS))
        p.start()
        processes.append(p)

    # ==========================================
    # 🚀 PYTORCH LOADED STRICTLY AFTER WORKERS
    # ==========================================
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    # 🚀 IMPORT MODEL FROM model.py
    from model import PyTorchAdvancedBlokusModel

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔥 Using device for testing: {device}", flush=True)

    pt_path = "blokus_expert_latest.pt"
    if not os.path.exists(pt_path):
        print(f"❌ Waiting for train_alphazero.py to output a .pt model first.", flush=True)
        return

    print(f"Warming up PyTorch for {pt_path}...", flush=True)
    model = PyTorchAdvancedBlokusModel(num_blocks=BLOCKS, filters=FILTERS).to(device)
    
    checkpoint = torch.load(pt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    is_mac = sys.platform == "darwin"
    
    # 🚀 SETUP: Enable Triton Fusion and CUDA Graphs for the Testing loop
    if not is_mac and hasattr(torch, "compile"):
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 64
        
        print(f"🔥 Wrapping model with torch.compile (Triton Fusion + CUDA Graphs Enabled)...", flush=True)
        fast_infer_model = torch.compile(
            model, 
            fullgraph=True,
            dynamic=False,  
            options={
                "max_autotune": True, 
                "triton.cudagraphs": True
            }
        )
    else:
        fast_infer_model = model

    # 🚀 JIT COMPILE AND GRAPH RECORDING WARMUP
    print(f"🔥 Pre-compiling Power-of-2 Buckets (64 to {2 ** MAX_ORDER}) into System RAM...", flush=True)
    fast_infer_model.eval()
    with torch.no_grad():
        for p in range(6, MAX_ORDER + 1):
            dummy = torch.zeros((2 ** p, 20, 20, 8), dtype=torch.float32, device=device)
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _ = fast_infer_model(dummy)
            else:
                _ = fast_infer_model(dummy)
    print("✅ All dynamic graphs successfully compiled and cached!", flush=True)

    print("="*60, flush=True)
    print(f"DISTRIBUTED PYTORCH BENCHMARK (BLOKUS - {num_games} GAMES)", flush=True)
    print("="*60, flush=True)

    start_time = time.time()
    
    test_inference_server(parent_conns, fast_infer_model, device, shared_counter, num_games, shared_data_bases, TOTAL_THREADS)

    total_aw, total_sw, total_t, total_diff = 0, 0, 0, 0
    for _ in range(NUM_WORKERS):
        aw, sw, t, diff = result_queue.get()
        total_aw += aw; total_sw += sw; total_t += t; total_diff += diff

    for p in processes: p.join()
    elapsed = time.time() - start_time
    avg_diff = total_diff / num_games

    print("\n" + "="*60, flush=True)
    print(f"FINAL BENCHMARK RESULTS (Completed in {elapsed:.1f}s)", flush=True)
    print("="*60, flush=True)
    print(f"Expert Bot Wins: {total_aw} ({(total_aw/num_games)*100:.1f}%)", flush=True)
    print(f"BotPlayer Wins:  {total_sw} ({(total_sw/num_games)*100:.1f}%)", flush=True)
    print(f"Ties:            {total_t} ({(total_t/num_games)*100:.1f}%)", flush=True)
    print(f"Avg Point Diff:  {'+' if avg_diff > 0 else ''}{avg_diff:.1f} pts per game for Expert Bot", flush=True)
    print("="*60, flush=True)

if __name__ == "__main__":
    mp.freeze_support()
    test_model()
