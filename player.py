import random
from helper import SHAPES, BOARD_SIZE, is_valid_move, rotate_shape, flip_shape

class BotPlayer:
    def get_play(self, board, color_id, available_shapes, is_first_move):
        # Create a copy of the available shapes and shuffle to add randomness
        shapes_to_try = list(available_shapes)
        random.shuffle(shapes_to_try) 
        
        for shape_name in shapes_to_try:
            base_coords = SHAPES[shape_name]
            
            current_coords = base_coords
            for flip_state in range(2):
                if flip_state == 1:
                    current_coords = flip_shape(base_coords)
                    
                for rot_state in range(4):
                    current_coords = rotate_shape(current_coords)
                    
                    for r in range(BOARD_SIZE):
                        for c in range(BOARD_SIZE):
                            shifted_coords = [(r + dr, c + dc) for dr, dc in current_coords]
                            
                            if is_valid_move(board, color_id, shifted_coords, is_first_move):
                                return shape_name, shifted_coords
                                
        return None, None
