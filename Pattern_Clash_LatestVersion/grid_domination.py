import numpy as np

class GridDominationGameAI:
    def __init__(self):
        self.grid = [["." for _ in range(5)] for _ in range(5)]
        self.scores = {"P1": 0, "P2": 0}
        self.current_player = "P1"
        self.pattern_scores = {
            "3-line": 5,    # Horizontal, vertical, or diagonal line of 3
            "2x2-square": 10, # Ahmed: 4 points should give greater than `3` points
            "diagonal": 7
        }
        self.used_cells = set()  # To keep track of cells used in any shape
        self.start = 1
        self.hash = []
        self.hashf = 0
        self.grid_dim = 5
        self.max_depth = 4
        # Ahmed: Variables
        self.con_leverage = [5, 50, 500]
        self.diagonal_leverage = [7, 70, 700]
        self.square_leverage = [1, 10, 100, 1000]

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
        player_reset_code = '1' if self.current_player == 'P1' else '2'
 
        # horizontal and vertical
        if self.check_cons(row, col, player_code):
            self.scores[self.current_player] += self.pattern_scores['3-line']
            remove_list = self.rest_con(row, col, player_code)
            self.reset_mark(remove_list, player_code)
        if self.check_dig(row, col, player_code):
            self.scores[self.current_player] += self.pattern_scores['diagonal']
            remove_list = self.reset_dia(row, col, player_code)
            self.reset_mark(remove_list, player_code)
        if self.squ(row, col, player_code):
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
                row, col = self.get_best_move(temp)
                self.make_move(row, col)
                print(f"AI plays at {chr(col + ord('A'))}{row + 1}")

        self.print_grid()
        winner = self.get_winner()
        print(f"Winner is: {winner}")
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
    def min(self, grid2, depth, alpha, beta):

        min_heu = float("inf")
        row, col = -1, -1

        # Base case: if depth is zero or the game is over, evaluate the grid
        # Ahmed: If depth is `5` and less than 5 places are remaning we will reach a point where
        #        board is complete and we want to right again, get `heuristic` directly if board completed
        if depth <= 0 or self.empty_cells_num(grid2) == 0 or (self.is_game_over() and self.empty_cells_num(grid2)!=0):
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

        # Ahmed: If depth is `5` and less than 5 places are remaning we will reach a point where
        #        board is complete and we want to right again, get `heuristic` directly if board completed
        if depth <= 0 or self.empty_cells_num(grid2) == 0 or (self.is_game_over() and self.empty_cells_num(grid2)!=0):
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
        
        # Ahmed: Throwing error if no place in Board
        if self.empty_cells_num(grid2) == 0:
            raise Exception("Board is complete")
        

        # Dynamic depth adjustment based on remaining empty cells
        # Ahmed: Comment Always do `5` Depth
        # if self.empty_cells_num(grid2) > 10:
        #     self.max_depth = 3
        # elif self.empty_cells_num(grid2) > 5:
        #     self.max_depth = 4
        # else:
        #     self.max_depth = 5

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

        # Ahmed: Should not return -1, -1
        if (best_row == -1 or best_col == -1):
            while (True):
                row = np.random.randint(0, 4);
                col = np.random.randint(0, 4);
                if self.is_valid_move_minimax(grid2, r, c):
                    return row, col
        return best_row, best_col
    
##############################################################################

########### Ahmed Functions #############
    def evaluate_cons(self, gridx, player):
        weight = 0
        for col in range(3):
            for row in range(5):
                p1, p2, p3 = gridx[row][col], gridx[row][col + 1], gridx[row][col + 2]
                count_points = [p1, p2, p3].count(player)
                count_empty = [p1, p2, p3].count('.')

                if (count_points + count_empty == 3):
                    weight += self.square_leverage[count_points - 1] if count_points > 0 else 0


                

        for col in range(5):
            for row in range(3):
                p1, p2, p3 = gridx[row][col], gridx[row + 1][col], gridx[row + 2][col]
                count_points = [p1, p2, p3].count(player)

                count_empty = [p1, p2, p3].count('.')

                if (count_points + count_empty == 3):
                    weight += self.square_leverage[count_points - 1] if count_points > 0 else 0

        return weight

    def evaluate_dig(self, gridx, player):
        weight = 0
        for col in range(3):
            for row in range(3):
                p1, p2, p3 = gridx[row][col], gridx[row + 1][col + 1], gridx[row + 2][col + 2]
                count_points = [p1, p2, p3].count(player)
                count_empty = [p1, p2, p3].count('.')

                if (count_points + count_empty == 3):
                    weight += self.square_leverage[count_points - 1] if count_points > 0 else 0

        return weight

    def evaluate_square(self, gridx, player):
        weight = 0
        for col in range(4):
            for row in range(4):
                p1, p2, p3, p4 = gridx[row][col], gridx[row + 1][col], gridx[row][col + 1], gridx[row + 1][col + 1]
                count_points = [p1, p2, p3, p4].count(player)
                count_empty = [p1, p2, p3].count('.')

                if (count_points + count_empty == 4):
                    weight += self.square_leverage[count_points - 1] if count_points > 0 else 0
        return weight

    def heuristic(self, gridx):
        weight_X = 0
        weight_O = 0
        weight_X += self.evaluate_cons(gridx, 'X')
        weight_X += self.evaluate_dig(gridx, 'X')
        weight_X += self.evaluate_square(gridx, 'X')
        weight_O += self.evaluate_cons(gridx, 'O')
        weight_O += self.evaluate_dig(gridx, 'O')
        weight_O += self.evaluate_square(gridx, '0')

        return weight_O - weight_X



# Run the game with AI
if __name__ == "__main__":
    game = GridDominationGameAI()
    game.play()
