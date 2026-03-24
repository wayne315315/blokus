import os
import time
import random
import multiprocessing as mp
import multiprocessing.connection 
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

from helper import BOARD_SIZE, SHAPES, is_valid_move, rotate_shape, flip_shape
from tf_alphazero_bot import BlokusAlphaZeroBot, create_policy_network

# ==============================================================================
# STANDARD GREEDY BOT
# ==============================================================================
class RandomGreedyBot:
    def get_play(self, board, color_id, inventories, first_moves, pass_count=0):
        inv_key = str(color_id) if str(color_id) in inventories else color_id
        available_shapes = inventories[inv_key]
        is_first_move = first_moves[color_id]
        
        shapes_to_try = list(available_shapes)
        random.shuffle(shapes_to_try) 
        
        for shape_name in shapes_to_try:
            base_coords = SHAPES[shape_name]
            current_coords = base_coords
            for flip_state in range(2):
                if flip_state == 1: current_coords = flip_shape(base_coords)
                for rot_state in range(4):
                    current_coords = rotate_shape(current_coords)
                    for r in range(BOARD_SIZE):
                        for c in range(BOARD_SIZE):
                            shifted_coords = [(r + dr, c + dc) for dr, dc in current_coords]
                            if is_valid_move(board, color_id, shifted_coords, is_first_move):
                                return shape_name, shifted_coords
                                
        return None, None

# ==============================================================================
# GPU INFERENCE SERVER
# ==============================================================================
def gpu_inference_server(conns, pol_net=None, fast_pol_infer=None):
    FIXED_BATCH = 8192
    
    if fast_pol_infer is None:
        @tf.function(input_signature=[tf.TensorSpec(shape=(FIXED_BATCH, 20, 20, 6), dtype=tf.float16)], jit_compile=True)
        def fast_pol_infer_internal(batch):
            return pol_net(batch, training=False)
        fast_pol_infer = fast_pol_infer_internal

        print("Compiling XLA Spatial Conv2D Kernel (This takes a few seconds)...", flush=True)
        _ = fast_pol_infer(tf.zeros((FIXED_BATCH, 20, 20, 6), dtype=tf.float16))
        print("XLA Compilation Complete! Benchmark Engine Armed.", flush=True)

    active_conns = list(conns)
    MIN_BATCH_SIZE = 256
    MAX_WAIT_TIME = 0.05

    pol_buffer = np.zeros((FIXED_BATCH, 20, 20, 6), dtype=np.float16)
    pol_pipes, pol_lens = [], []
    pol_cursor = 0
    last_fire_time = time.time()

    while active_conns:
        ready_conns = []
        for i in range(0, len(active_conns), 500):
            chunk = active_conns[i : i+500]
            ready_conns.extend(multiprocessing.connection.wait(chunk, timeout=0.001))
        
        for conn in ready_conns:
            try:
                msg = conn.recv()
                if msg == "DONE":
                    active_conns.remove(conn)
                else:
                    is_policy, inputs = msg
                    if is_policy:  
                        length = len(inputs)
                        if pol_cursor + length > FIXED_BATCH:
                            tensor = tf.convert_to_tensor(pol_buffer, dtype=tf.float16)
                            preds = fast_pol_infer(tensor).numpy()
                            idx = 0
                            for c, l in zip(pol_pipes, pol_lens):
                                c.send(preds[idx : idx + l])
                                idx += l
                            pol_pipes, pol_lens = [], []
                            pol_cursor = 0
                            last_fire_time = time.time()
                            
                        pol_buffer[pol_cursor : pol_cursor + length] = inputs
                        pol_pipes.append(conn)
                        pol_lens.append(length)
                        pol_cursor += length
            except EOFError:
                if conn in active_conns:
                    active_conns.remove(conn)
                    
        time_waiting = time.time() - last_fire_time
        
        if pol_cursor >= MIN_BATCH_SIZE or (time_waiting > MAX_WAIT_TIME and pol_cursor > 0):
            tensor = tf.convert_to_tensor(pol_buffer, dtype=tf.float16)
            preds = fast_pol_infer(tensor).numpy()
            
            idx = 0
            for c, length in zip(pol_pipes, pol_lens):
                c.send(preds[idx : idx + length])
                idx += length
                
            pol_pipes, pol_lens = [], []
            pol_cursor = 0
            last_fire_time = time.time()

# ==============================================================================
# MATCH SIMULATION THREAD
# ==============================================================================
def _thread_test_games(num_games, conn, play_as_first):
    az_bot = BlokusAlphaZeroBot("AlphaZero", pipe=conn, is_training=False)
    std_bot = RandomGreedyBot()
    
    az_wins, std_wins, ties = 0, 0, 0
    az_score_diff = 0
    
    for _ in range(num_games):
        board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        inventories = { 1: list(SHAPES.keys()), 2: list(SHAPES.keys()), 3: list(SHAPES.keys()), 4: list(SHAPES.keys()) }
        first_moves = {1: True, 2: True, 3: True, 4: True}
        
        if play_as_first:
            color_map = {1: az_bot, 2: std_bot, 3: az_bot, 4: std_bot}
        else:
            color_map = {1: std_bot, 2: az_bot, 3: std_bot, 4: az_bot}
            
        current_color = 1
        pass_count = 0

        while pass_count < 4:
            active_bot = color_map[current_color]
            
            shape_name, coords = active_bot.get_play(board, current_color, inventories, first_moves, pass_count)
            
            if shape_name is None:
                pass_count += 1
            else:
                for r, c in coords:
                    board[r][c] = current_color
                inventories[current_color].remove(shape_name)
                first_moves[current_color] = False
                pass_count = 0
                
            current_color = (current_color % 4) + 1

        def get_score(col_id):
            return sum(-int(shape.split('_')[0]) for shape in inventories[col_id])

        if play_as_first:
            az_score = get_score(1) + get_score(3)
            std_score = get_score(2) + get_score(4)
        else:
            std_score = get_score(1) + get_score(3)
            az_score = get_score(2) + get_score(4)

        az_score_diff += (az_score - std_score)

        if az_score > std_score:
            az_wins += 1
        elif std_score > az_score:
            std_wins += 1
        else:
            ties += 1
            
    return az_wins, std_wins, ties, az_score_diff

def distributed_test_worker(num_games, conns, result_queue, worker_idx):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    threads_count = len(conns)
    base_games = num_games // threads_count
    remainder = num_games % threads_count
    thread_tasks = [base_games + (1 if i < remainder else 0) for i in range(threads_count)]

    results = []
    with ThreadPoolExecutor(max_workers=threads_count) as executor:
        futures = []
        for i in range(threads_count):
            if thread_tasks[i] > 0:
                play_first = (i % 2 == 0)
                futures.append(executor.submit(_thread_test_games, thread_tasks[i], conns[i], play_first))
        for future in futures:
            results.append(future.result())

    total_az_wins, total_std_wins, total_ties, total_diff = 0, 0, 0, 0
    for aw, sw, t, diff in results:
        total_az_wins += aw
        total_std_wins += sw
        total_ties += t
        total_diff += diff

    for conn in conns:
        conn.send("DONE") 

    result_queue.put((total_az_wins, total_std_wins, total_ties, total_diff))

def test_model(num_games=100, policy_path="tf_policy_model.keras", num_workers=None, threads_per_worker=10):
    print("="*60)
    print(f"DISTRIBUTED FLOAT16 GPU BENCHMARK (BLOKUS - {num_games} GAMES)")
    print("="*60)
    
    if num_workers is None: num_workers = max(1, mp.cpu_count() - 1)
        
    if num_games < num_workers * threads_per_worker:
        num_workers = max(1, num_games // threads_per_worker)
        if num_workers == 0:
            num_workers = 1
            threads_per_worker = num_games
        
    total_threads = num_workers * threads_per_worker
    print(f"Spawning {num_workers} processes x {threads_per_worker} threads ({total_threads} virtual actors)...")
    start_time = time.time()
    
    try:
        shared_pol_net = tf.keras.models.load_model(policy_path, compile=False)
        print(f"Successfully loaded full model '{policy_path}'")
    except:
        print(f"Warning: Could not load '{policy_path}'. Using random weights. This will perform poorly.")
        shared_pol_net = create_policy_network()
        
    ctx = mp.get_context('spawn')
    
    base_games = num_games // num_workers
    remainder = num_games % num_workers
    test_tasks = [base_games + (1 if i < remainder else 0) for i in range(num_workers)]
    
    pipes = [ctx.Pipe() for _ in range(total_threads)]
    parent_conns = [p[0] for p in pipes]
    child_conns = [p[1] for p in pipes]
    
    child_conn_chunks = [child_conns[i * threads_per_worker : (i + 1) * threads_per_worker] for i in range(num_workers)]
    
    result_queue = ctx.Queue()

    processes = []
    for i, task_count in enumerate(test_tasks):
        if task_count > 0:
            p = ctx.Process(target=distributed_test_worker, args=(task_count, child_conn_chunks[i], result_queue, i))
            p.start()
            processes.append(p)

    gpu_inference_server(parent_conns, shared_pol_net)

    total_az_wins, total_std_wins, total_ties, total_diff = 0, 0, 0, 0
    for _ in range(len(processes)):
        aw, sw, t, diff = result_queue.get()
        total_az_wins += aw
        total_std_wins += sw
        total_ties += t
        total_diff += diff

    for p in processes: p.join()
        
    elapsed = time.time() - start_time
    avg_diff = total_diff / num_games

    print("\n" + "="*60)
    print(f"FINAL BENCHMARK RESULTS (Completed in {elapsed:.1f}s)")
    print("="*60)
    print(f"Total Games Played:  {num_games}")
    print(f"AlphaZero Bot Wins:  {total_az_wins} ({(total_az_wins/num_games)*100:.1f}%)")
    print(f"Standard Bot Wins:   {total_std_wins} ({(total_std_wins/num_games)*100:.1f}%)")
    print(f"Ties:                {total_ties} ({(total_ties/num_games)*100:.1f}%)")
    print(f"Avg Point Diff:      {'+' if avg_diff > 0 else ''}{avg_diff:.1f} pts per game for AlphaZero")
    print("="*60)

if __name__ == "__main__":
    mp.freeze_support()
    test_model(num_games=200, threads_per_worker=10)
