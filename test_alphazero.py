import os
import time
import multiprocessing as mp
import multiprocessing.connection 
import ctypes
import numpy as np
import tensorflow as tf

from helper import BOARD_SIZE, SHAPES
from player import BotPlayer  

MAX_CAPACITY = 2048 

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

def distributed_test_worker(num_games, conn, result_queue, worker_idx, shared_counter, shared_data_bases, num_workers):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    from tf_alphazero_bot import ExpertBlokusBot
    
    shared_states = np.ctypeslib.as_array(shared_data_bases[0].get_obj()).reshape((num_workers, MAX_CAPACITY, 20, 20, 6))
    shared_values = np.ctypeslib.as_array(shared_data_bases[1].get_obj()).reshape((num_workers, MAX_CAPACITY))
    shared_scores = np.ctypeslib.as_array(shared_data_bases[2].get_obj()).reshape((num_workers, MAX_CAPACITY))
    
    shared_data = (worker_idx, shared_states, shared_values, shared_scores)
    
    adv_bot = ExpertBlokusBot(pipe=conn, shared_data=shared_data, is_training=False)
    std_bot_1 = BotPlayer() 
    std_bot_2 = BotPlayer()

    az_wins, std_wins, ties, az_score_diff = 0, 0, 0, 0
    for i in range(num_games):
        play_as_first = (i % 2 == 0)
        aw, sw, t, diff = play_test_game(adv_bot, std_bot_1, std_bot_2, play_as_first)
        az_wins += aw; std_wins += sw; ties += t; az_score_diff += diff
        
        with shared_counter.get_lock():
            shared_counter.value += 1

    conn.send("DONE") 
    result_queue.put([az_wins, std_wins, ties, az_score_diff])

def test_inference_server(conns, fast_infer, shared_counter, total_games, shared_data_bases, num_workers):
    shared_states = np.ctypeslib.as_array(shared_data_bases[0].get_obj()).reshape((num_workers, MAX_CAPACITY, 20, 20, 6))
    shared_values = np.ctypeslib.as_array(shared_data_bases[1].get_obj()).reshape((num_workers, MAX_CAPACITY))
    shared_scores = np.ctypeslib.as_array(shared_data_bases[2].get_obj()).reshape((num_workers, MAX_CAPACITY))

    active_conns = list(conns)
    conn_to_id = {conns[i]: i for i in range(num_workers)}
    
    ready_indices, batch_sizes, ready_pipes = [], [], []
    total_states_queued = 0
    last_fire_time = time.time()

    MIN_BATCH_SIZE = 1024 
    MAX_WAIT_TIME = 0.05 
    CHUNK_SIZE = 2048 

    while active_conns:
        readable = multiprocessing.connection.wait(active_conns, timeout=0.002)
        
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

        if total_states_queued >= MIN_BATCH_SIZE or (total_states_queued > 0 and time.time() - last_fire_time > MAX_WAIT_TIME):
            
            flat_states = [shared_states[w_id, :size] for w_id, size in zip(ready_indices, batch_sizes)]
            batch_tensor = np.concatenate(flat_states, axis=0)
            actual_size = batch_tensor.shape[0]
            
            v_preds, sc_preds = [], []
            
            for i in range(0, actual_size, CHUNK_SIZE):
                chunk = batch_tensor[i : i + CHUNK_SIZE]
                curr_size = chunk.shape[0]
                
                # 🚀 DYNAMIC BUCKETING
                pad_target = 1 << (curr_size - 1).bit_length()
                
                if curr_size < pad_target:
                    pad = np.zeros((pad_target - curr_size, 20, 20, 6), dtype=np.float32)
                    chunk_padded = np.concatenate([chunk, pad], axis=0)
                else:
                    chunk_padded = chunk
                    
                preds = fast_infer(chunk_padded)
                
                v_preds.append(preds[0].numpy().flatten()[:curr_size])
                sc_preds.append(preds[1].numpy().flatten()[:curr_size])
                
            v_numpy = np.concatenate(v_preds)
            sc_numpy = np.concatenate(sc_preds)
            
            cursor = 0
            for w_id, size in zip(ready_indices, batch_sizes):
                shared_values[w_id, :size] = v_numpy[cursor:cursor+size]
                shared_scores[w_id, :size] = sc_numpy[cursor:cursor+size]
                cursor += size

            games_remaining = total_games - shared_counter.value
            print(f"⚡ GPU INFERENCE | States Evaluated: {total_states_queued:<4} | Games Remaining: {games_remaining:<4} ", end='\r', flush=True)

            for pipe in ready_pipes: pipe.send(True)

            ready_indices, batch_sizes, ready_pipes = [], [], []
            total_states_queued = 0
            last_fire_time = time.time()
            
    print("\n✅ All test games completed!")

def test_model(num_games=100):
    print("="*60)
    print(f"DISTRIBUTED TENSORFLOW BENCHMARK (BLOKUS - {num_games} GAMES)")
    print("="*60)
    
    keras_path = "blokus_expert_latest.keras"
    if not os.path.exists(keras_path):
        fallback = "blokus_expert_v18.keras"
        if os.path.exists(fallback): keras_path = fallback
        elif os.path.exists("blokus_expert_v0.keras"): keras_path = "blokus_expert_v0.keras"
        else:
            print(f"❌ Waiting for train_alphazero.py to output a model first.")
            return

    gpus = tf.config.list_physical_devices('GPU')
    if gpus: tf.config.experimental.set_memory_growth(gpus[0], True)

    print(f"Warming up TensorFlow Graph Compiler for {keras_path}...")
    model = tf.keras.models.load_model(keras_path, compile=False)
    
    @tf.function(reduce_retracing=True)
    def fast_infer(batch_tensor): 
        return model(batch_tensor, training=False)

    print("🔥 Pre-compiling Power-of-2 XLA Buckets (1, 2, 4... 2048) into System RAM...")
    for p in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        _ = fast_infer(tf.zeros((p, 20, 20, 6), dtype=tf.float32))
    print("✅ All 12 dynamic graphs successfully compiled and cached!")
    
    num_workers = min(31, mp.cpu_count() - 1) 
    print(f"Starting {num_workers} CPU Process Workers...")
    
    start_time = time.time()
    ctx = mp.get_context('spawn')
    
    shared_counter = ctx.Value('i', 0)
    
    shared_states_base = mp.Array(ctypes.c_float, num_workers * MAX_CAPACITY * 20 * 20 * 6)
    shared_values_base = mp.Array(ctypes.c_float, num_workers * MAX_CAPACITY)
    shared_scores_base = mp.Array(ctypes.c_float, num_workers * MAX_CAPACITY)
    shared_data_bases = (shared_states_base, shared_values_base, shared_scores_base)

    pipes = [ctx.Pipe() for _ in range(num_workers)]
    parent_conns, child_conns = [p[0] for p in pipes], [p[1] for p in pipes]
    
    result_queue = ctx.Queue()
    processes = []
    
    base_games = num_games // num_workers
    tasks = [base_games + (1 if i < num_games % num_workers else 0) for i in range(num_workers)]
    
    for i in range(num_workers):
        if tasks[i] > 0:
            p = ctx.Process(target=distributed_test_worker, args=(tasks[i], child_conns[i], result_queue, i, shared_counter, shared_data_bases, num_workers))
            p.start()
            processes.append(p)

    test_inference_server(parent_conns[:len(processes)], fast_infer, shared_counter, num_games, shared_data_bases, num_workers)

    total_aw, total_sw, total_t, total_diff = 0, 0, 0, 0
    for _ in range(len(processes)):
        aw, sw, t, diff = result_queue.get()
        total_aw += aw; total_sw += sw; total_t += t; total_diff += diff

    for p in processes: p.join()
    elapsed = time.time() - start_time
    avg_diff = total_diff / num_games

    print("\n" + "="*60)
    print(f"FINAL BENCHMARK RESULTS (Completed in {elapsed:.1f}s)")
    print("="*60)
    print(f"Expert Bot Wins: {total_aw} ({(total_aw/num_games)*100:.1f}%)")
    print(f"BotPlayer Wins:  {total_sw} ({(total_sw/num_games)*100:.1f}%)")
    print(f"Ties:            {total_t} ({(total_t/num_games)*100:.1f}%)")
    print(f"Avg Point Diff:  {'+' if avg_diff > 0 else ''}{avg_diff:.1f} pts per game for Expert Bot")
    print("="*60)

if __name__ == "__main__":
    mp.freeze_support()
    test_model(num_games=100)