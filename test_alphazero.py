import os
import time
import multiprocessing as mp
import multiprocessing.connection 
import ctypes
import numpy as np

from helper import BOARD_SIZE, SHAPES
from player import BotPlayer  

MAX_ORDER = 10
_NUM_CPUS = mp.cpu_count()
NUM_WORKERS = min(31, _NUM_CPUS - 1 if _NUM_CPUS > 1 else 1)
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

def distributed_test_worker(games_per_worker, conn, result_queue, worker_idx, shared_counter, shared_data_bases, num_workers):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    from tf_alphazero_bot import ExpertBlokusBot
    
    # 🚀 TENSOR CORE OPTIMIZATION: Channels bumped to 8
    shared_states = np.ctypeslib.as_array(shared_data_bases[0].get_obj()).reshape((num_workers, MAX_CAPACITY, 20, 20, 8))
    shared_values = np.ctypeslib.as_array(shared_data_bases[1].get_obj()).reshape((num_workers, MAX_CAPACITY))
    shared_scores = np.ctypeslib.as_array(shared_data_bases[2].get_obj()).reshape((num_workers, MAX_CAPACITY))
    
    bot = ExpertBlokusBot(pipe=conn, shared_data=(worker_idx, shared_states, shared_values, shared_scores), is_training=False)
    std_bot_1 = BotPlayer() 
    std_bot_2 = BotPlayer()

    az_wins, std_wins, ties, az_score_diff = 0, 0, 0, 0
    for i in range(games_per_worker):
        play_as_first = (i % 2 == 0)
        aw, sw, t, diff = play_test_game(bot, std_bot_1, std_bot_2, play_as_first)
        az_wins += aw; std_wins += sw; ties += t; az_score_diff += diff
        with shared_counter.get_lock(): shared_counter.value += 1

    conn.send("DONE") 
    result_queue.put([az_wins, std_wins, ties, az_score_diff])

def test_inference_server(conns, fast_infer, shared_counter, total_games, shared_data_bases, num_workers):
    import tensorflow as tf
    
    # 🚀 TENSOR CORE OPTIMIZATION: Channels bumped to 8
    shared_states = np.ctypeslib.as_array(shared_data_bases[0].get_obj()).reshape((num_workers, MAX_CAPACITY, 20, 20, 8))
    shared_values = np.ctypeslib.as_array(shared_data_bases[1].get_obj()).reshape((num_workers, MAX_CAPACITY))
    shared_scores = np.ctypeslib.as_array(shared_data_bases[2].get_obj()).reshape((num_workers, MAX_CAPACITY))

    active_conns = list(conns)
    conn_to_id = {conns[i]: i for i in range(num_workers)}
    
    ready_indices, batch_sizes, ready_pipes = [], [], []
    total_states_queued = 0
    last_print_time = time.time()
    
    CHUNK_SIZE = 2 ** MAX_ORDER
    MIN_BATCH_SIZE = 64

    print("🔥 Allocating Static Server Buffer in System RAM...", flush=True)
    MAX_POSSIBLE_BATCH = num_workers * MAX_CAPACITY
    # 🚀 TENSOR CORE OPTIMIZATION: Channels bumped to 8
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
            v_preds, sc_preds = [], []
            
            tensor_cursor = 0
            rem = actual_size
            
            while rem > 0:
                curr_size = min(rem, CHUNK_SIZE)
                
                pad_target = 1 << (curr_size - 1).bit_length() if curr_size > 0 else 0
                if pad_target < MIN_BATCH_SIZE: 
                    pad_target = MIN_BATCH_SIZE
                
                chunk = batch_tensor[tensor_cursor : tensor_cursor + curr_size]
                
                if pad_target > curr_size:
                    # 🚀 TENSOR CORE OPTIMIZATION: Channels bumped to 8
                    pad_array = np.zeros((pad_target - curr_size, 20, 20, 8), dtype=np.float32)
                    chunk_padded = np.concatenate([chunk, pad_array], axis=0)
                else:
                    chunk_padded = chunk
                    
                preds = fast_infer(chunk_padded)
                v_preds.append(preds[0].numpy().flatten()[:curr_size])
                sc_preds.append(preds[1].numpy().flatten()[:curr_size])
                
                tensor_cursor += curr_size
                rem -= curr_size
                
            t_infer = (time.time() - t1) * 1000 
            t_per_sample = (t_infer / actual_size) if actual_size > 0 else 0.0
            
            t2 = time.time()
            v_numpy = np.concatenate(v_preds)
            sc_numpy = np.concatenate(sc_preds)
            
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
                      f"Copy: {t_copy:>5.1f}ms | "
                      f"Infer: {t_infer:>5.1f}ms ({t_per_sample:>4.4f}ms/st) | "
                      f"Write: {t_write:>5.1f}ms | "
                      f"Remaining: {games_remaining:<4} ", end='\r', flush=True)
                last_print_time = curr_time

            ready_indices, batch_sizes, ready_pipes = [], [], []
            total_states_queued = 0
            
    print("\n✅ All test games completed!")

def test_model(num_games=124):
    import tensorflow as tf
    
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

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
    
    @tf.function(reduce_retracing=True, jit_compile=True)
    def fast_infer(batch_tensor): 
        return model(batch_tensor, training=False)

    print(f"🔥 Pre-compiling Power-of-2 XLA Buckets (64 to {MAX_CAPACITY}) into System RAM...")
    for p in [64, 128, 256, 512, 1024, 2048]:
        if p > 2 ** MAX_ORDER: break
        # 🚀 TENSOR CORE OPTIMIZATION: Channels bumped to 8
        _ = fast_infer(tf.zeros((p, 20, 20, 8), dtype=tf.float32))
    print("✅ All dynamic graphs successfully compiled and cached!")

    print("="*60)
    print(f"DISTRIBUTED TENSORFLOW BENCHMARK (BLOKUS - {num_games} GAMES)")
    print("="*60)

    start_time = time.time()
    ctx = mp.get_context('spawn')
    
    shared_counter = ctx.Value('i', 0)
    # 🚀 TENSOR CORE OPTIMIZATION: Channels bumped to 8
    shared_states_base = mp.Array(ctypes.c_float, NUM_WORKERS * MAX_CAPACITY * 20 * 20 * 8)
    shared_values_base = mp.Array(ctypes.c_float, NUM_WORKERS * MAX_CAPACITY)
    shared_scores_base = mp.Array(ctypes.c_float, NUM_WORKERS * MAX_CAPACITY)
    shared_data_bases = (shared_states_base, shared_values_base, shared_scores_base)

    pipes = [ctx.Pipe() for _ in range(NUM_WORKERS)]
    parent_conns, child_conns = [p[0] for p in pipes], [p[1] for p in pipes]
    
    result_queue = ctx.Queue()
    processes = []
    
    games_per_worker = max(1, num_games // NUM_WORKERS)
    
    for i in range(NUM_WORKERS):
        p = ctx.Process(target=distributed_test_worker, args=(games_per_worker, child_conns[i], result_queue, i, shared_counter, shared_data_bases, NUM_WORKERS))
        p.start()
        processes.append(p)

    test_inference_server(parent_conns[:len(processes)], fast_infer, shared_counter, num_games, shared_data_bases, NUM_WORKERS)

    total_aw, total_sw, total_t, total_diff = 0, 0, 0, 0
    for _ in range(NUM_WORKERS):
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
    test_model()