import numpy as np

class GridDominationGameAI:
    def __init__(self):
        self.grid = [["." for _ in range(5)] for _ in range(5)]
        self.scores = {"P1": 0, "P2": 0}
        self.current_player = "P1"
        self.pattern_scores = {
            "3-line": 5,    # Horizontal, vertical, or diagonal line of 3
            "2x2-square": 6,
            "diagonal": 7
        }
        self.used_cells = set()  # To keep track of cells used in any shape
        self.start = 1
        self.hash = []
        self.hashf = 0
        self.grid_dim = 5
        self.max_depth = 4
# 333

    def print_grid(self):
        print("  A  B  C  D  E")
        for i, row in enumerate(self.grid, 1):
            print(f"{i} {'  '.join(row)}")
################################################################

    def is_valid_move(self, row, col):
        return 0 <= row < 5 and 0 <= col < 5 and self.grid[row][col] == "." and self.grid[row][col] != 1 and self.grid[row][col] != 2
################################################################

    def get_winner(self):
        if self.scores["P1"] > self.scores["P2"]:
            return "P1"
        elif self.scores["P1"] < self.scores["P2"]:
            return "P2"
        return "Tie"
################################################################

    def make_move(self, row, col):
        if not self.is_valid_move(row, col):
            return False
        self.grid[row][col] = "X" if self.current_player == "P1" else "O"
        # Add this line to call the score update
        self.update_score(row, col)
        self.current_player = "P1" if self.current_player == "P2" else "P2"
        return True
################################################################

    def update_score(self, row, col):
        """
        Update scores for both players based on pattern recognition.

        Args:
            row (int): Row of the most recently placed cell
            col (int): Column of the most recently placed cell
        """
        player_code = 'X' if self.current_player == 'P1' else 'O'

        # horizontal and vertical
        if self.check_cons(row, col, player_code):
            self.scores[self.current_player] += self.pattern_scores['3-line']
            remove_list = self.rest_con(row, col, player_code)
            self.reset_mark(remove_list, player_code)
        if self.check_dig(row, col, player_code):                   # diagonal
            self.scores[self.current_player] += self.pattern_scores['diagonal']
            remove_list = self.reset_dia(row, col, player_code)
            self.reset_mark(remove_list, player_code)
        if self.reset_squ(row, col, player_code)[0]:  # square
            self.scores[self.current_player] += self.pattern_scores['2x2-square']
            remove_list = self.reset_squ(row, col, player_code)[1]
            self.reset_mark(remove_list, player_code)

################################################################
    def reset_mark(self, lists, code):
        self.hashf = 1
        self.hash = lists
        t = '1' if code == "X" else '2'
        for row, col in lists:
            self.grid[row][col] = t
################################################################

    def rest_con(self, row, col, code):
        points = []
        if row-2 >= 0:
            if self.grid[row-2][col] == code and self.grid[row-1][col] == code:
                points.append((row-2, col))
                points.append((row-1, col))
                points.append((row, col))
        if row-1 >= 0 and row+1 < 5:
            if self.grid[row-1][col] == code and self.grid[row+1][col] == code:
                points.append((row-1, col))
                points.append((row+1, col))
                points.append((row, col))

        if row+2 < 5:
            if self.grid[row+1][col] == code and self.grid[row+2][col] == code:
                points.append((row+1, col))
                points.append((row+2, col))
                points.append((row, col))

        if col-2 >= 0:
            if self.grid[row][col-2] == code and self.grid[row][col-1] == code:
                points.append((row, col-2))
                points.append((row, col-1))
                points.append((row, col))

        if col-1 >= 0 and col+1 < 5:
            if self.grid[row][col-1] == code and self.grid[row][col+1] == code:
                points.append((row, col-1))
                points.append((row, col+1))
                points.append((row, col))

        if col+2 < 5:
            if self.grid[row][col+1] == code and self.grid[row][col+2] == code:
                points.append((row, col+1))
                points.append((row, col+2))
                points.append((row, col))
        return points
################################################################

    def reset_squ(self, row, col, code):
        square_shape = [
            [(0, 0), (1, 0), (0, 1), (1, 1)],
            [(0, 0), (0, 1), (-1, 0), (-1, 1)],
            [(0, 0), (1, 0), (1, -1), (0, -1)],
            [(0, 0), (0, -1), (-1, 0), (-1, -1)],
        ]
        points = []
        for square in square_shape:
            cnt = 0
            for r, c in square:
                fr = r+row
                fc = c+col
                if 0 <= fr < 5 and 0 <= fc < 5 and self.grid[fr][fc] == code:
                    cnt += 1
                    points.append((fr, fc))
                else:
                    points = []
                    break
                if cnt == 4:
                    return 1, points
        return 0, list()


################################################################


    def reset_dia(self, x, y, code):
        points = []
        if x-2 >= 0 and y+2 < 5:
            if self.grid[x-2][y+2] == code and self.grid[x-1][y+1] == code:
                points.append((x-2, y+2))
                points.append((x-1, y+1))
                points.append((x, y))
        if x-2 >= 0 and y-2 >= 0:
            if self.grid[x-2][y-2] == code and self.grid[x-1][y-1] == code:
                points.append((x-2, y-2))
                points.append((x-1, y-1))
                points.append((x, y))
        if x+2 < 5 and y+2 < 5:
            if self.grid[x+2][y+2] == code and self.grid[x+1][y+1] == code:
                points.append((x+2, y+2))
                points.append((x+1, y+1))
                points.append((x, y))
        if x+2 < 5 and y-2 >= 0:
            if self.grid[x+2][y-2] == code and self.grid[x+1][y-1] == code:
                points.append((x+2, y-2))
                points.append((x+1, y-1))
                points.append((x, y))
        if x-1 >= 0 and x+1 < 5 and y+1 < 5 and y-1 >= 0:
            if self.grid[x-1][y+1] == code and self.grid[x+1][y-1] == code:
                points.append((x-1, y+1))
                points.append((x+1, y-1))
                points.append((x, y))
        if x-1 >= 0 and x+1 < 5 and y+1 < 5 and y-1 >= 0:
            if self.grid[x-1][y-1] == code and self.grid[x+1][y+1] == code:
                points.append((x-1, y-1))
                points.append((x+1, y+1))
                points.append((x, y))
        return points
################################################################

    def check_dig(self, x, y, code):
        flag = 0
        if x-2 >= 0 and y+2 < 5:
            if self.grid[x-2][y+2] == code and self.grid[x-1][y+1] == code:
                flag = 1
        if x-2 >= 0 and y-2 >= 0:
            if self.grid[x-2][y-2] == code and self.grid[x-1][y-1] == code:
                flag = 1
        if x+2 < 5 and y+2 < 5:
            if self.grid[x+2][y+2] == code and self.grid[x+1][y+1] == code:
                flag = 1
        if x+2 < 5 and y-2 >= 0:
            if self.grid[x+2][y-2] == code and self.grid[x+1][y-1] == code:
                flag = 1
        if x-1 >= 0 and x+1 < 5 and y+1 < 5 and y-1 >= 0:
            if self.grid[x-1][y+1] == code and self.grid[x+1][y-1] == code:
                flag = 1
        if x-1 >= 0 and x+1 < 5 and y+1 < 5 and y-1 >= 0:
            if self.grid[x-1][y-1] == code and self.grid[x+1][y+1] == code:
                flag = 1
        return flag
################################################################

    def check_cons(self, x, y, code):
        flag = 0
        if x+2 < 5:
            if self.grid[x+2][y] == code and self.grid[x+1][y] == code:
                flag = 1
        if x-2 >= 0:
            if self.grid[x-2][y] == code and self.grid[x-1][y] == code:
                flag = 1
        if y+2 < 5:
            if self.grid[x][y+2] == code and self.grid[x][y+1] == code:
                flag = 1
        if y-2 >= 0:
            if self.grid[x][y-2] == code and self.grid[x][y-1] == code:
                flag = 1
        if x-1 >= 0 and x+1 < 5:
            if self.grid[x-1][y] == code and self.grid[x+1][y] == code:
                flag = 1
        if y-1 >= 0 and y+1 < 5:
            if self.grid[x][y-1] == code and self.grid[x][y+1] == code:
                flag = 1
        return flag
################################################################

    def squ(self, row, col, code):
        flag = 0
        square_shape = [
            [(0, 0), (1, 0), (0, 1), (1, 1)],
            [(0, 0), (0, 1), (-1, 0), (-1, 1)],
            [(0, 0), (1, 0), (1, -1), (0, -1)],
            [(0, 0), (0, -1), (-1, 0), (-1, -1)],
        ]
        for square in square_shape:
            cnt = 0
            for r, c in square:
                fr = r+row
                fc = c+col
                if 0 <= fr < 5 and 0 <= fc < 5 and self.grid[fr][fc] == code:
                    cnt += 1
                else:
                    break
                if cnt == 4:
                    flag = 1
        return flag

################################################################
    def is_game_over(self):
        if self.scores["P1"] >= 20 or self.scores["P2"] >= 20:
            return True
        return all(cell != "." for row in self.grid for cell in row)

################################################################
    def play(self):
        print("Welcome to Grid Domination: Pattern Clash with AI!")
        while not self.is_game_over():
            self.print_grid()
            print(f"Current scores: {self.scores}")

            if self.current_player == "P1":
                move = input("Enter your move (row col): ").upper()
                col = move[0]  # First character is the column letter
                # Second character is the row number (adjusted for 0 index)
                row = int(move[1]) - 1

                # Convert column letter (A, B, C, etc.) to index (0, 1, 2, etc.)
                col_index = ord(col) - ord('A')

                if not self.make_move(row, col_index):
                    print("Invalid move. Try again.")
                    continue
            else:
                print("AI is making its move...")
                temp = [row[:] for row in self.grid] 
                row, col = self. get_best_move(temp)
                self.make_move(row, col)
                print(f"AI plays at {chr(col + ord('A'))}{row + 1}")

        self.print_grid()
        winner = self.get_winner()
        print(f"Winner is: {winner}")
################################################################

    def check_dig_minmax(self,gridx, x, y, code):
        counter = 0
        if x+2 < 5 and y+2 < 5:
            if (gridx[x+2][y+2] == code or gridx[x+2][y+2] == '.') and (gridx[x+1][y+1] == code or gridx[x+1][y+1] == '.'):
                counter += 1
        if x+2 < 5 and y-2 >= 0:
            if (gridx[x+2][y-2] == code or gridx[x+2][y-2] == '.') and (gridx[x+1][y-1] == code or gridx[x+1][y-1] == '.'):
                counter += 1
        return counter

################################################################
    def check_cons_minmax(self,gridx, x, y, code):
        counter = 0
        if x+2 < 5:
            if (gridx[x+2][y] == code or gridx[x+2][y] == '.') and (gridx[x+1][y] == code or gridx[x+1][y] == '.'):
                counter += 1
        if y+2 < 5:
            if (gridx[x][y+2] == code or gridx[x][y+2] == '.') and (gridx[x][y+1] == code or gridx[x][y+1] == '.'):
                counter += 1
        ## for a winning move 
        if x+2 < 5:
            if (self.grid[x+2][y] == code) and (self.grid[x+1][y] == code ):
                if code =="O":
                    return float("inf")  
                else:
                    return float("-inf")  
        if y+2 < 5:
            if (self.grid[x][y+2] == code ) and (self.grid[x][y+1] == code ):
               if code =="O":
                   return float("inf")  
               else:
                   return float("-inf") 
        
        return counter
################################################################

    def squ_minmax(self,gridx, row, col, code):
        counter = 0
        square_shape = [
            [(0, 0), (1, 0), (0, 1), (1, 1)]
        ]
        for square in square_shape:
            cnt = 0
            for r, c in square:
                fr = r+row
                fc = c+col
                if 0 <= fr < 5 and 0 <= fc < 5 and (gridx[fr][fc] == code or gridx[fr][fc] == '.'):
                    cnt += 1
                else:
                    break
                if cnt == 4:
                    counter += 1
        return counter
################################################################

    def empty_cells_num(self, grid):
        counter = 0
        for r in range(self.grid_dim):
            for c in range(self.grid_dim):
                if grid[r][c] == '.':
                    counter += 1
        return counter

################################################################
    def is_valid_move_minimax(self,gridx, row, col):
        return 0 <= row < 5 and 0 <= col < 5 and gridx[row][col] == "." and gridx[row][col] != 1 and gridx[row][col] != 2

################################################################
    def is_winning_move(self, gridx, code):
        for r in range(self.grid_dim):
            for c in range(self.grid_dim):
                if gridx[r][c] == code:
                    if self.check_cons_minmax(gridx, r, c, code) == float("inf") or \
                        self.check_dig_minmax(gridx, r, c, code) == float("inf") or \
                        self.squ_minmax(gridx, r, c, code) > 0:
                        return True
        return False


################################################################
    def heuristic(self,gridx):
        # The evaluation function heuristic
        counter_X = 0
        counter_O = 0
        if self.is_winning_move(gridx, "O"):
            return float("inf")  # AI wins
        if self.is_winning_move(gridx, "X"):
            return float("-inf")  # Opponent wins
        for r in range(self.grid_dim):
            for c in range(self.grid_dim):
                if self.is_valid_move(r, c):
                    counter_X += self.check_cons_minmax(gridx,r, c, "X")
                    counter_X += self.check_dig_minmax(gridx,r, c, "X")
                    counter_X += self.squ_minmax(gridx,r, c, "X")

                    counter_O += self.check_cons_minmax(gridx,r, c, "O")
                    counter_O += self.check_dig_minmax(gridx,r, c, "O")
                    counter_O += self.squ_minmax(gridx,r, c, "O")

        return counter_O - counter_X

################################################################
    def min(self, grid2, depth, alpha, beta):

        min_heu = float("inf")
        row, col = -1, -1

        # Base case: if depth is zero or the game is over, evaluate the grid
        if depth <= 0 or self.is_game_over() and self.empty_cells_num(grid2)!=0:
            return self.heuristic(grid2), row, col

        for r in range(self.grid_dim):
            for c in range(self.grid_dim):
                if self.is_valid_move_minimax(grid2, r, c):
                    # Simulate the move
                    grid2[r][c] = "X"
                    result, _, _ = self.max(grid2, depth - 1, alpha, beta)
                    grid2[r][c] = "."  # Undo the move
                    
                    if result < min_heu:
                        min_heu, row, col = result, r, c
                    beta = min(beta, min_heu)
                    
                    # Alpha-beta pruning
                    if beta <= alpha:
                        break

        return min_heu, row, col
    
####################################################################
    def max(self, grid2, depth, alpha, beta):
        max_heu = float("-inf")
        row, col = -1, -1
        if depth <= 0 or self.is_game_over() and self.empty_cells_num(grid2)!=0:
            return self.heuristic(grid2), row, col
        for r in range(self.grid_dim):
            for c in range(self.grid_dim):
                if self.is_valid_move_minimax(grid2, r, c):
                    grid2[r][c] = "O"
                    result, _, _ = self.min(grid2, depth - 1, alpha, beta)
                    grid2[r][c] = "."
                    if result > max_heu:
                        max_heu, row, col = result, r, c
                    alpha = max(alpha, max_heu)
                    if beta <= alpha:
                        break
        return max_heu, row, col
    
###########################################################################

    def get_best_move(self, grid2):
        
        # Dynamic depth adjustment based on remaining empty cells
        if self.empty_cells_num(grid2) > 10:
            self.max_depth = 3
        elif self.empty_cells_num(grid2) > 5:
            self.max_depth = 4
        else:
            self.max_depth = 5

        # Initialize variables for the best move
        best_value = float("-inf")
        best_row, best_col = -1, -1
        alpha=float("-inf")
        beta=float("inf")
        # Call the max function to get the best move for the AI
        for r in range(self.grid_dim):
            for c in range(self.grid_dim):
                if self.is_valid_move_minimax(grid2, r, c):
                    grid2[r][c] = "O"  # Simulate AI's move
                    value, _, _ = self.min(grid2, self.max_depth - 1,alpha,beta)
                    grid2[r][c] = "."  # Undo the move

                    if value > best_value:
                        best_value, best_row, best_col = value, r, c

        return best_row, best_col
    
##############################################################################


# Run the game with AI
if __name__ == "__main__":
    game = GridDominationGameAI()
    game.play()
