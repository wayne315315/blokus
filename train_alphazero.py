import os
import time
import math
import pickle
import numpy as np
import multiprocessing as mp
import multiprocessing.connection 
import threading
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from tensorflow.keras import optimizers, mixed_precision

mixed_precision.set_global_policy('mixed_float16')

from helper import BOARD_SIZE, SHAPES
from tf_alphazero_bot import BlokusAlphaZeroBot, create_value_network, create_policy_network

def add_to_buffer(buffer_x, buffer_y, new_x, new_y, current_size, max_size):
    n = len(new_x)
    if n == 0: return current_size
    if current_size + n <= max_size:
        buffer_x[current_size:current_size+n] = new_x
        buffer_y[current_size:current_size+n] = new_y
        return current_size + n
    else:
        rem = max_size - current_size
        if rem > 0:
            buffer_x[current_size:max_size] = new_x[:rem]
            buffer_y[current_size:max_size] = new_y[:rem]
        n_left = n - rem
        if n_left > 0:
            replace_idx = np.random.randint(0, max_size, size=n_left)
            buffer_x[replace_idx] = new_x[rem:]
            buffer_y[replace_idx] = new_y[rem:]
        return max_size

def gpu_inference_server(conns, fast_val_infer, fast_pol_infer):
    # --- TUNED FOR NVIDIA L4 (24GB VRAM) ---
    FIXED_BATCH = 12288 
    MIN_BATCH_SIZE = 8192
    MAX_WAIT_TIME = 0.01
    # ---------------------------------------

    active_conns = list(conns)

    val_buffer = np.zeros((FIXED_BATCH, 20, 20, 6), dtype=np.float16)
    pol_buffer = np.zeros((FIXED_BATCH, 20, 20, 6), dtype=np.float16)
    
    val_pipes, val_lens = [], []
    pol_pipes, pol_lens = [], []
    
    val_cursor, pol_cursor = 0, 0
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
                    length = len(inputs)
                    
                    if is_policy:
                        # Policy Buffer Full - Fire instantly
                        if pol_cursor + length > FIXED_BATCH:
                            print(f"[DEBUG] Pol Batch Fired (Buffer Full): {pol_cursor}")
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
                    else:
                        # Value Buffer Full - Fire instantly
                        if val_cursor + length > FIXED_BATCH:
                            print(f"[DEBUG] Val Batch Fired (Buffer Full): {val_cursor}")
                            tensor = tf.convert_to_tensor(val_buffer, dtype=tf.float16)
                            preds = fast_val_infer(tensor).numpy()
                            idx = 0
                            for c, l in zip(val_pipes, val_lens):
                                c.send(preds[idx : idx + l])
                                idx += l
                            val_pipes, val_lens = [], []
                            val_cursor = 0
                            last_fire_time = time.time()
                            
                        val_buffer[val_cursor : val_cursor + length] = inputs
                        val_pipes.append(conn)
                        val_lens.append(length)
                        val_cursor += length
            except EOFError:
                if conn in active_conns:
                    active_conns.remove(conn)

        time_waiting = time.time() - last_fire_time

        # Time/Minimum Threshold Firing Checks
        if val_cursor >= MIN_BATCH_SIZE or (time_waiting > MAX_WAIT_TIME and val_cursor > 0):
            print(f"[DEBUG] Val batch size: {val_cursor}")
            tensor = tf.convert_to_tensor(val_buffer, dtype=tf.float16)
            preds = fast_val_infer(tensor).numpy()
            idx = 0
            for c, length in zip(val_pipes, val_lens):
                c.send(preds[idx : idx + length])
                idx += length
            val_pipes, val_lens = [], []
            val_cursor = 0
            last_fire_time = time.time()
                
        if pol_cursor >= MIN_BATCH_SIZE or (time_waiting > MAX_WAIT_TIME and pol_cursor > 0):
            print(f"[DEBUG] Pol batch size: {pol_cursor}")
            tensor = tf.convert_to_tensor(pol_buffer, dtype=tf.float16)
            preds = fast_pol_infer(tensor).numpy()
            idx = 0
            for c, length in zip(pol_pipes, pol_lens):
                c.send(preds[idx : idx + length])
                idx += length
            pol_pipes, pol_lens = [], []
            pol_cursor = 0
            last_fire_time = time.time()


def _thread_simulate_games(num_games, conn, current_episode, total_episodes):
    
    # COSINE ANNEALING: Bounces exploration between 0.35 and 0.02
    progress = min(1.0, current_episode / total_episodes)
    current_exploration = 0.02 + 0.5 * (0.35 - 0.02) * (1 + math.cos(math.pi * progress))

    bot1 = BlokusAlphaZeroBot("P1 (Blue/Red)", pipe=conn, is_training=True, exploration_rate=current_exploration)
    bot2 = BlokusAlphaZeroBot("P2 (Yellow/Green)", pipe=conn, is_training=True, exploration_rate=current_exploration)
    
    color_map = {1: bot1, 2: bot2, 3: bot1, 4: bot2}

    accumulated_train_x, accumulated_train_y = [], []
    policy_x, policy_y = [], []
    p1_wins, p2_wins = 0, 0
    game_lengths = []

    for _ in range(num_games):
        bot1.clear_memory()
        bot2.clear_memory()

        board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        inventories = { 1: list(SHAPES.keys()), 2: list(SHAPES.keys()), 3: list(SHAPES.keys()), 4: list(SHAPES.keys()) }
        first_moves = {1: True, 2: True, 3: True, 4: True}
        
        current_color = 1
        pass_count = 0
        turns_played = 0

        while pass_count < 4:
            active_bot = color_map[current_color]
            
            # FULL MCTS Call
            shape_name, coords = active_bot.get_play(board, current_color, inventories, first_moves, pass_count)
            
            if shape_name is None:
                pass_count += 1
            else:
                for r, c in coords:
                    board[r][c] = current_color
                inventories[current_color].remove(shape_name)
                first_moves[current_color] = False
                pass_count = 0
                turns_played += 1
                
            current_color = (current_color % 4) + 1

        def get_score(col_id):
            return sum(-int(shape.split('_')[0]) for shape in inventories[col_id])

        p1_score = get_score(1) + get_score(3)
        p2_score = get_score(2) + get_score(4)

        if p1_score > p2_score:
            p1_wins += 1
            p1_reward, p2_reward = 1.0, -1.0
        elif p2_score > p1_score:
            p2_wins += 1
            p1_reward, p2_reward = -1.0, 1.0
        else:
            p1_reward, p2_reward = 0.0, 0.0

        game_lengths.append(turns_played)

        for step in bot1.episode_memory:
            accumulated_train_x.append(step['inputs'][step['chosen_index']])
            accumulated_train_y.append(p1_reward) 
            
        for mem in bot1.policy_memory:
            policy_x.append(mem[0])
            policy_y.append(mem[1])

        for step in bot2.episode_memory:
            accumulated_train_x.append(step['inputs'][step['chosen_index']])
            accumulated_train_y.append(p2_reward)
            
        for mem in bot2.policy_memory:
            policy_x.append(mem[0])
            policy_y.append(mem[1])

    return (
        np.array(accumulated_train_x, dtype=np.float16), 
        np.array(accumulated_train_y, dtype=np.float32), 
        np.array(policy_x, dtype=np.float16), 
        np.array(policy_y, dtype=np.float32), 
        p1_wins, p2_wins, game_lengths
    )

def worker_generate_batch(num_games, conns, result_queue, current_episode, total_episodes):
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
                futures.append(executor.submit(_thread_simulate_games, thread_tasks[i], conns[i], current_episode, total_episodes))
        for future in futures:
            results.append(future.result())

    accumulated_train_x, accumulated_train_y = [], []
    policy_x, policy_y = [], []
    p1_wins, p2_wins = 0, 0
    game_lengths = []

    for r in results:
        val_x, val_y, pol_x, pol_y, p1w, p2w, gl = r
        accumulated_train_x.extend(val_x)
        accumulated_train_y.extend(val_y)
        policy_x.extend(pol_x)
        policy_y.extend(pol_y)
        p1_wins += p1w
        p2_wins += p2w
        game_lengths.extend(gl)

    for conn in conns: conn.send("DONE") 
    result_queue.put((
        np.array(accumulated_train_x, dtype=np.float16), np.array(accumulated_train_y, dtype=np.float32), 
        np.array(policy_x, dtype=np.float16), np.array(policy_y, dtype=np.float32), 
        p1_wins, p2_wins, game_lengths
    ))

def train_self_play(total_episodes=200000, batch_size=2048, val_path='tf_value_model.keras', pol_path='tf_policy_model.keras', buffer_path='az_replay_buffers.pkl', num_workers=None, threads_per_worker=10, episodes_per_update=50):
    print("="*60, flush=True)
    print("INITIALIZING 2D SPATIAL ALPHAZERO PIPELINE (BLOKUS)", flush=True)
    print("="*60, flush=True)
    
    if num_workers is None: num_workers = max(1, mp.cpu_count() - 1)
    total_threads = num_workers * threads_per_worker

    try:
        shared_val_net = tf.keras.models.load_model(val_path)
        shared_pol_net = tf.keras.models.load_model(pol_path)
        print(f"Loaded existing FULL models from .keras files.", flush=True)
    except Exception:
        print("Starting with fresh network weights and optimizers.", flush=True)
        shared_val_net = create_value_network()
        shared_pol_net = create_policy_network()
        
    optimizer_val = optimizers.Adam(learning_rate=1e-4)
    optimizer_pol = optimizers.Adam(learning_rate=1e-4)

    shared_val_net.compile(optimizer=optimizer_val, loss='mse', jit_compile=True)
    shared_pol_net.compile(optimizer=optimizer_pol, loss='mse', jit_compile=True) 

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(batch_size, 20, 20, 6), dtype=tf.float16),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.float32)
    ], jit_compile=True)
    def train_val_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            preds = shared_val_net(x_batch, training=True)
            loss = tf.reduce_mean(tf.square(y_batch - tf.squeeze(preds)))
        grads = tape.gradient(loss, shared_val_net.trainable_variables)
        optimizer_val.apply_gradients(zip(grads, shared_val_net.trainable_variables))
        return loss

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(batch_size, 20, 20, 6), dtype=tf.float16),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.float32)
    ], jit_compile=True)
    def train_pol_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            preds = shared_pol_net(x_batch, training=True)
            loss = tf.reduce_mean(tf.square(y_batch - tf.squeeze(preds)))
        grads = tape.gradient(loss, shared_pol_net.trainable_variables)
        optimizer_pol.apply_gradients(zip(grads, shared_pol_net.trainable_variables))
        return loss

    # --- Match this to your tuned Server settings above ---
    FIXED_BATCH = 12288
    # ------------------------------------------------------

    @tf.function(input_signature=[tf.TensorSpec(shape=(FIXED_BATCH, 20, 20, 6), dtype=tf.float16)], jit_compile=True)
    def fast_val_infer(batch): return shared_val_net(batch, training=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=(FIXED_BATCH, 20, 20, 6), dtype=tf.float16)], jit_compile=True)
    def fast_pol_infer(batch): return shared_pol_net(batch, training=False)

    print("Compiling XLA Spatial Conv2D Kernels...", flush=True)
    _ = fast_val_infer(tf.zeros((FIXED_BATCH, 20, 20, 6), dtype=tf.float16))
    _ = fast_pol_infer(tf.zeros((FIXED_BATCH, 20, 20, 6), dtype=tf.float16))
    _ = train_val_step(tf.zeros((batch_size, 20, 20, 6), dtype=tf.float16), tf.zeros((batch_size,), dtype=tf.float32))
    _ = train_pol_step(tf.zeros((batch_size, 20, 20, 6), dtype=tf.float16), tf.zeros((batch_size,), dtype=tf.float32))
    print("XLA Compilation Complete! Inference Engine Armed.", flush=True)

    metrics = {'p1_wins': 0, 'p2_wins': 0, 'game_lengths': [], 'val_losses': [], 'pol_losses': []}
    start_time = time.time()
    ctx = mp.get_context('spawn')
    
    MAX_BUFFER_SIZE = 100000
    replay_val_x = np.zeros((MAX_BUFFER_SIZE, 20, 20, 6), dtype=np.float16)
    replay_val_y = np.zeros((MAX_BUFFER_SIZE,), dtype=np.float32)
    replay_pol_x = np.zeros((MAX_BUFFER_SIZE, 20, 20, 6), dtype=np.float16)
    replay_pol_y = np.zeros((MAX_BUFFER_SIZE,), dtype=np.float32)
    
    replay_val_size = 0
    replay_pol_size = 0
    episodes_completed = 0
    total_games_generated = 0

    if os.path.exists(buffer_path):
        with open(buffer_path, 'rb') as f:
            save_data = pickle.load(f)
            loaded_val_x = save_data['val_x']
            replay_val_size = len(loaded_val_x)
            replay_val_x[:replay_val_size] = loaded_val_x
            replay_val_y[:replay_val_size] = save_data['val_y']
            
            loaded_pol_x = save_data['pol_x']
            replay_pol_size = len(loaded_pol_x)
            replay_pol_x[:replay_pol_size] = loaded_pol_x
            replay_pol_y[:replay_pol_size] = save_data['pol_y']
            
            episodes_completed = save_data['episodes_completed']
            total_games_generated = save_data['total_games_generated']
        print(f"Restored {replay_val_size} states.", flush=True)

    while episodes_completed < total_episodes:
        base_games = episodes_per_update // num_workers
        remainder = episodes_per_update % num_workers
        worker_tasks = [base_games + (1 if i < remainder else 0) for i in range(num_workers)]

        pipes = [ctx.Pipe() for _ in range(total_threads)]
        parent_conns = [p[0] for p in pipes]
        child_conns = [p[1] for p in pipes]
        
        child_conn_chunks = [child_conns[i * threads_per_worker : (i + 1) * threads_per_worker] for i in range(num_workers)]
        result_queue = ctx.Queue()

        print("  [Phase 1] Simulation (CPU+GPU Infer)... \n", end="", flush=True)
        t_sim_start = time.time()
        
        processes = []
        for i, task_count in enumerate(worker_tasks):
            if task_count > 0:
                p = ctx.Process(target=worker_generate_batch, args=(task_count, child_conn_chunks[i], result_queue, episodes_completed, total_episodes))
                p.start()
                processes.append(p)

        gpu_inference_server(parent_conns, fast_val_infer, fast_pol_infer)
        
        d_sim = time.time() - t_sim_start
        print(f"\n  Done in {d_sim:.2f}s", flush=True)
        
        print("  [Phase 2] Buffer Collection...          ", end="", flush=True)
        t_col_start = time.time()
        
        for _ in range(len(processes)):
            val_x, val_y, pol_x, pol_y, p1w, p2w, gl = result_queue.get()
            metrics['p1_wins'] += p1w
            metrics['p2_wins'] += p2w
            metrics['game_lengths'].extend(gl)
            total_games_generated += len(val_x)
            
            replay_val_size = add_to_buffer(replay_val_x, replay_val_y, val_x, val_y, replay_val_size, MAX_BUFFER_SIZE)
            replay_pol_size = add_to_buffer(replay_pol_x, replay_pol_y, pol_x, pol_y, replay_pol_size, MAX_BUFFER_SIZE)

        for p in processes: p.join()
        
        d_col = time.time() - t_col_start
        print(f"Done in {d_col:.2f}s", flush=True)

        if replay_val_size > 0 and replay_pol_size > 0:
            print("  [Phase 3] Model Fitting (GPU Train)...  ", end="", flush=True)
            t_fit_start = time.time()
            
            target_batch_size = 16000
            
            sample_val_size = min(target_batch_size, replay_val_size)
            sample_val_size = (sample_val_size // batch_size) * batch_size
            if sample_val_size == 0: sample_val_size = batch_size
            idx_val = np.random.choice(replay_val_size, sample_val_size, replace=False)
            train_val_x, train_val_y = replay_val_x[idx_val], replay_val_y[idx_val]
            
            sample_pol_size = min(target_batch_size, replay_pol_size)
            sample_pol_size = (sample_pol_size // batch_size) * batch_size
            if sample_pol_size == 0: sample_pol_size = batch_size
            idx_pol = np.random.choice(replay_pol_size, sample_pol_size, replace=False)
            train_pol_x, train_pol_y = replay_pol_x[idx_pol], replay_pol_y[idx_pol]
            
            val_loss_sum = 0.0
            for i in range(0, sample_val_size, batch_size):
                end = i + batch_size
                val_loss_sum += float(train_val_step(train_val_x[i:end], train_val_y[i:end]))
                
            pol_loss_sum = 0.0
            for i in range(0, sample_pol_size, batch_size):
                end = i + batch_size
                pol_loss_sum += float(train_pol_step(train_pol_x[i:end], train_pol_y[i:end]))
                
            metrics['val_losses'].append(val_loss_sum / max(1, sample_val_size // batch_size))
            metrics['pol_losses'].append(pol_loss_sum / max(1, sample_pol_size // batch_size))
            
            d_fit = time.time() - t_fit_start
            print(f"Done in {d_fit:.2f}s", flush=True)
            
            shared_val_net.save(val_path)
            shared_pol_net.save(pol_path)
            episodes_completed += episodes_per_update

        avg_val = metrics['val_losses'][-1] if metrics['val_losses'] else 0.0
        avg_pol = metrics['pol_losses'][-1] if metrics['pol_losses'] else 0.0
        avg_len = np.mean(metrics['game_lengths'][-episodes_per_update:])
        
        elapsed = time.time() - start_time
        print("\n" + "="*60, flush=True)
        print(f"Episodes {episodes_completed}/{total_episodes} | Total Time: {elapsed:.1f}s", flush=True)
        print(f"  -> Win Rate: P1 ({metrics['p1_wins']}) vs P2 ({metrics['p2_wins']})", flush=True)
        print(f"  -> Avg Game Length: {avg_len:.1f} turns", flush=True)
        print(f"  -> Val Loss: {avg_val:.4f} | Pol Loss: {avg_pol:.4f}", flush=True)
        print("-" * 60, flush=True)
        metrics['p1_wins'], metrics['p2_wins'] = 0, 0

if __name__ == "__main__":
    mp.freeze_support()
    # Increase threads_per_worker to 16 for better L4 saturation
    train_self_play(total_episodes=50000, episodes_per_update=50, batch_size=2048, threads_per_worker=32)
