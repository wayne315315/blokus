import numpy as np

BOARD_SIZE = 20

SHAPES = {
    "1_I1": [(0,0)],
    "2_I2": [(0,0), (0,1)],
    "3_I3": [(0,0), (0,1), (0,2)],
    "3_V3": [(0,0), (1,0), (1,1)],
    "4_I4": [(0,0), (0,1), (0,2), (0,3)],
    "4_L4": [(0,0), (1,0), (2,0), (2,1)],
    "4_T4": [(0,0), (0,1), (0,2), (1,1)],
    "4_O4": [(0,0), (0,1), (1,0), (1,1)],
    "4_Z4": [(0,0), (0,1), (1,1), (1,2)],
    "5_I5": [(0,0), (0,1), (0,2), (0,3), (0,4)],
    "5_L5": [(0,0), (1,0), (2,0), (3,0), (3,1)],
    "5_Y5": [(0,0), (0,1), (0,2), (0,3), (1,1)],
    "5_N5": [(0,0), (0,1), (1,1), (1,2), (1,3)],
    "5_P5": [(0,0), (0,1), (1,0), (1,1), (2,0)],
    "5_U5": [(0,0), (0,1), (1,0), (1,2), (0,2)],
    "5_V5": [(0,0), (1,0), (2,0), (2,1), (2,2)],
    "5_Z5": [(0,0), (0,1), (1,1), (2,1), (2,2)],
    "5_T5": [(0,0), (0,1), (0,2), (1,1), (2,1)],
    "5_F5": [(0,1), (0,2), (1,0), (1,1), (2,1)],
    "5_W5": [(0,0), (1,0), (1,1), (2,1), (2,2)],
    "5_X5": [(0,1), (1,0), (1,1), (1,2), (2,1)]
}

def rotate_shape(coords):
    return [(c, -r) for r, c in coords]

def flip_shape(coords):
    return [(r, -c) for r, c in coords]

def is_valid_move(board, color_id, coords, is_first_move):
    has_corner_contact = False
    
    for r, c in coords:
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE): return False
        if board[r][c] != 0: return False 
            
        # Check Edges (Cannot touch SAME color side-by-side)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if board[nr][nc] == color_id: return False
                    
        # Check Corners (Must touch at least one SAME color corner)
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if board[nr][nc] == color_id: has_corner_contact = True

    # First Move Constraint maps each color to its specific corner
    if is_first_move:
        starting_corners = {
            1: (0, 0),      # Blue Top-Left
            2: (0, 19),     # Yellow Top-Right
            3: (19, 19),    # Red Bottom-Right (Diagonal to Blue)
            4: (19, 0)      # Green Bottom-Left (Diagonal to Yellow)
        }
        req_corner = starting_corners[color_id]
        return any((r, c) == req_corner for r, c in coords)
        
    return has_corner_contact