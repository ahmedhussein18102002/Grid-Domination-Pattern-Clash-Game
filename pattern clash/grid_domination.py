import numpy as np
class GridDominationGameAI:
    def __init__(self):
        self.grid = [["." for _ in range(5)] for _ in range(5)]
        self.scores = {"P1": 0, "P2": 0}
        self.current_player = "P1"
        self.pattern_scores = {
            "3-line": 5,    # Horizontal, vertical, or diagonal line of 3
            "2x2-square": 6,
            "L-shape": 7,
            "full-row-col": 10,
            "diagonal": 7
        }
        self.used_cells = set()  # To keep track of cells used in any shape
        self.start=1
        self.hash=[] 
        self.hashf=0
################################################################333
    def print_grid(self):
        print("  A  B  C  D  E")
        for i, row in enumerate(self.grid, 1):
            print(f"{i} {'  '.join(row)}")
################################################################
    def is_valid_move(self, row, col):
        return 0 <= row < 5 and 0 <= col < 5 and self.grid[row][col] == "." and self.grid[row][col] !=1 and self.grid[row][col] !=2 
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
          player_code='X' if self.current_player =='P1' else 'O'

          if self.check_cons(row,col,player_code):                 # horizontal and vertical
             self.scores[self.current_player] += self.pattern_scores['3-line']
             remove_list = self.rest_con(row,col,player_code)
             self.reset_mark(remove_list,player_code)
          if self.check_dig(row,col,player_code):                   # diagonal
              self.scores[self.current_player] += self.pattern_scores['diagonal']
              remove_list = self.reset_dia(row,col,player_code)
              self.reset_mark(remove_list,player_code)
          if self.reset_squ(row,col,player_code)[0]:                         #square
              self.scores[self.current_player] += self.pattern_scores['2x2-square']
              remove_list = self.reset_squ(row,col,player_code)[1]
              self.reset_mark(remove_list,player_code)
          
################################################################
    def reset_mark(self,lists,code):
      self.hashf=1
      self.hash=lists
      t = '1' if code == "X" else '2'
      for row,col in lists:
        self.grid[row][col] = t
################################################################
    def rest_con(self,row,col,code):
      points=[]
      if row-2>=0:
        if self.grid[row-2][col] == code and self.grid[row-1][col] == code:
          points.append((row-2,col))
          points.append((row-1,col))
          points.append((row,col))
      if row-1>=0 and row+1 <5:
        if self.grid[row-1][col] == code and self.grid[row+1][col] == code:
          points.append((row-1,col))
          points.append((row+1,col))
          points.append((row,col))

      if row+2 <5:
        if self.grid[row+1][col] == code and self.grid[row+2][col] == code:
          points.append((row+1,col))
          points.append((row+2,col))
          points.append((row,col))

      if col-2>=0:
        if self.grid[row][col-2] == code and self.grid[row][col-1] == code:
          points.append((row,col-2))
          points.append((row,col-1))
          points.append((row,col))

      if col-1>=0 and col+1 <5:
        if self.grid[row][col-1] == code and self.grid[row][col+1] == code:
          points.append((row,col-1))
          points.append((row,col+1))
          points.append((row,col))

      if col+2 <5:
        if self.grid[row][col+1] == code and self.grid[row][col+2] == code:
          points.append((row,col+1))
          points.append((row+2,col+2))
          points.append((row,col))
      return points
################################################################
    def reset_squ (self,row,col,code):
      square_shape = [
              [(0, 0), (1, 0), (0, 1),(1,1)],
              [(0, 0), (0, 1), (-1, 0),(-1,1)],
              [(0, 0), (1, 0), (1, -1),(0,-1)],
              [(0, 0), (0, -1), (-1, 0),(-1,-1)],
          ]
      points=[]
      for square in square_shape:
        cnt=0
        for r,c in square:
          fr=r+row
          fc=c+col
          if 0<= fr <5 and 0<=fc <5 and self.grid[fr][fc]==code:
            cnt+=1
            points.append((fr,fc))
          else :
            points=[]
            break
          if cnt==4:
            return 1,points
      return 0,list()


################################################################
    def reset_dia(self,x,y,code):
        points=[]
        if x-2 >=0 and y+2 <5:
          if self.grid[x-2][y+2]==code and self.grid[x-1][y+1]==code:
            points.append((x-2,y+2))
            points.append((x-1,y+1))
            points.append((x,y))
        if x-2>=0 and y-2>=0:
          if self.grid[x-2][y-2]==code and self.grid[x-1][y-1]==code:
            points.append((x-2,y-2))
            points.append((x-1,y-1))
            points.append((x,y))
        if x+2 <5 and y+2 <5:
          if self.grid[x+2][y+2]==code and self.grid[x+1][y+1]==code:
            points.append((x+2,y+2))
            points.append((x+1,y+1))
            points.append((x,y))
        if x+2<5 and y-2>=0:
          if self.grid[x+2][y-2]==code and self.grid[x+1][y-1]==code:
            points.append((x+2,y-2))
            points.append((x+1,y-1))
            points.append((x,y))
        if x-1>=0 and x+1<5 and y+1 <5 and y-1 >=0:
          if self.grid[x-1][y+1]==code and self.grid[x+1][y-1]==code:
            points.append((x-1,y+1))
            points.append((x+1,y-1))
            points.append((x,y))
        if x-1>=0 and x+1<5 and y+1 <5 and y-1 >=0:
          if self.grid[x-1][y-1]==code and self.grid[x+1][y+1]==code:
            points.append((x-1,y-1))
            points.append((x+1,y+1))
            points.append((x,y))
        return points
################################################################
    def check_dig(self,x,y,code):
      flag=0
      if x-2 >=0 and y+2 <5:
        if self.grid[x-2][y+2]==code and self.grid[x-1][y+1]==code:
          flag=1
      if x-2>=0 and y-2>=0:
        if self.grid[x-2][y-2]==code and self.grid[x-1][y-1]==code:
          flag=1
      if x+2 <5 and y+2 <5:
        if self.grid[x+2][y+2]==code and self.grid[x+1][y+1]==code:
          flag=1
      if x+2<5 and y-2>=0:
        if self.grid[x+2][y-2]==code and self.grid[x+1][y-1]==code:
          flag=1
      if x-1>=0 and x+1<5 and y+1 <5 and y-1 >=0:
        if self.grid[x-1][y+1]==code and self.grid[x+1][y-1]==code:
          flag=1
      if x-1>=0 and x+1<5 and y+1 <5 and y-1 >=0:
        if self.grid[x-1][y-1]==code and self.grid[x+1][y+1]==code:
          flag=1
      return flag
################################################################
    def check_cons (self,x,y,code):
        flag=0
        if x+2 < 5:
          if self.grid[x+2][y]==code and self.grid[x+1][y]==code :
             flag=1
        if x-2 >= 0:
          if self.grid[x-2][y]==code and self.grid[x-1][y]==code :
            flag=1
        if y+2 < 5:
          if self.grid[x][y+2]==code and self.grid[x][y+1]==code :
            flag=1
        if y-2 >= 0:
          if self.grid[x][y-2]==code and self.grid[x][y-1]==code :
            flag=1
        if x-1 >=0 and x+1 <5:
          if self.grid[x-1][y]==code and self.grid[x+1][y]==code:
            flag=1
        if y-1 >=0 and y+1 <5:
          if self.grid[x][y-1]==code and self.grid[x][y+1]==code :
            flag=1
        return flag
################################################################
    def squ(self, row, col, code):
      flag = 0
      square_shape = [
              [(0, 0), (1, 0), (0, 1),(1,1)],
              [(0, 0), (0, 1), (-1, 0),(-1,1)],
              [(0, 0), (1, 0), (1, -1),(0,-1)],
              [(0, 0), (0, -1), (-1, 0),(-1,-1)],
          ]
      for square in square_shape:
        cnt=0
        for r,c in square:
          fr=r+row
          fc=c+col
          if 0<= fr <5 and 0<=fc <5 and self.grid[fr][fc]==code:
            cnt+=1
          else :
            break
          if cnt==4:
            flag=1
      return flag
################################################################
    
################################################################
    def best_diagonal_move(self, player='O'):
      n = len(self.grid)
      best_move = None
      min_moves = float('inf')
      score=7
      # Define directions for diagonals: [\ and /]
      directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

      for row in range(n):
          for col in range(n):
            if self.grid[row][col] ==player :
              for dr, dc in directions:
                  count = 1  # Include the current cell
                  empty_cell = None
                  for step in range(1,3):  # Check up to 2 more cells
                      r, c = row + dr * step, col + dc * step
                      if 0 <= r < n and 0 <= c < n and self.grid [r][c] !='X':
                          if self.grid[r][c] == player:
                              count += 1
                          elif self.grid[r][c] == '.':
                              if empty_cell is None:
                                  empty_cell = (r, c)
                      else:
                          # empty_cell=None
                          break
                  # if empty_cell is not None:
                  #   if self.check_dig(empty_cell[0],empty_cell[1]) ==0:
                  #     empty_cell= None
                  if count == 2 and empty_cell:  # Check for two player cells and one empty
                      if min_moves > 1:  # Update best move
                          min_moves = 1
                          best_move = empty_cell

                  elif count == 1 and empty_cell and not best_move:  # Check potential for one player cell
                      min_moves = 2
                      best_move = empty_cell
      return (best_move, min_moves,score) if best_move else (None, -1)
      
################################################################
    def check_HV(self, player='O'):
      n = 5
      best_move = None
      min_moves = float('inf')
      score=5
      useful = 1
      directions = [(-1,0),(1,0),(0,1),(0,-1)]
      for row in range(n):
        for col in range(n):
          if self.grid[row][col] ==player :
            for dr, dc in directions:
              count = 1  # Include the current cell
              empty_cell = None
              for step in range(1,3):  # Check up to 2 more cells
                  r, c = row + dr * step, col + dc * step
                  if 0 <= r < n and 0 <= c < n and self.grid [r][c] !='X':
                        if self.grid[r][c] == player:
                            count += 1
                        elif self.grid[r][c] == '.':
                            if empty_cell is None:
                                empty_cell = (r, c)

                  else:
                      break
              # if empty_cell is not None:
                  # if self.check_cons(empty_cell[0],empty_cell[1]) ==0:
                    # empty_cell= None
              if count == 2 and empty_cell:  # Check for two player cells and one empty
                    if min_moves > 1:  # Update best move
                        min_moves = 1
                        best_move = empty_cell

              elif count == 1 and empty_cell and not best_move:  # Check potential for one player cell
                    min_moves = 2
                    best_move = empty_cell
      return (best_move, min_moves,score) if best_move else (None, -1)
      
################################################################
    def check_best_move_v (self,player='O'):
      n= len(self.grid)
      best_move = None
      score = 10
      min_move = float('inf')
      empty_cell = None
      for row in range(n):
        for col in range(n):
          if self.grid[row][col] == player: # concsened of that point
            empty_cell=None
            for r in range(n):
              if self.grid[r][col] ==player or self.grid[r][col] == '.':
                if self.grid[r][col] == '.':
                  if empty_cell is None:
                    empty_cell = (r,col)
                  else: # another empty cell but we already have a cell
                    break 
              else : # this point never gonna have veritacla line
                break
            if empty_cell is not None: # we have a solution
              cnt =0
              flag=0
              for r in range(n):
                if self.grid[r][col] == player:
                  cnt += 1
                elif self.grid[r][col]=='X':
                  flag=1
                  break
              if min_move >(n-cnt) and flag==0:
                min_move= n-cnt
                best_move= empty_cell
      return (best_move,min_move,score) if best_move else (None,-1)    
################################################################
    def check_best_move_H (self,player='O'):
      n= len(self.grid)
      best_move = None
      score = 10
      min_move = float('inf')
      empty_cell = None
      for row in range(n):
        for col in range(n):
          if self.grid[row][col] == player: # concsened of that point
            empty_cell=None
            for c in range(n):
              if self.grid[row][c] ==player or self.grid[row][c] == '.':
                if self.grid[row][c] == '.':
                  if empty_cell is None:
                    empty_cell = (row,c)
                  else: # another empty cell but we already have a cell
                    break 
              else : # this point never gonna have veritacla line
                break
            if empty_cell is not None: # we have a solution
              cnt =0
              flag=0
              for c in range(n):
                if self.grid[row][c] == player:
                  cnt += 1
                elif self.grid[row][c]=='X':
                  flag=1
                  break
              if min_move >(n-cnt) and flag==0:
                min_move= n-cnt
                best_move= empty_cell
      return (best_move,min_move,score) if best_move else (None,-1) 
     
################################################################
    def check_square(self, player='O'):
          n = len(self.grid)
          best_move = None
          min_moves = float('inf')
          score = 6  # Slightly higher score than line patterns

          #  configurations (rotations)
          square_shape = [
              [(0, 0), (1, 0), (0, 1),(1,1)],
              [(0, 0), (0, 1), (-1, 0),(-1,1)],
              [(0, 0), (1, 0), (1, -1),(0,-1)],
              [(0, 0), (0, -1), (-1, 0),(-1,1)],
          ]

          for row in range(n):
              for col in range(n):
                if self.grid[row][col] ==player :
                  for shape in square_shape:
                      empty_cell = None
                      count_player_cells = 0

                      for r, c in shape:
                          check_row, check_col = row + r, col + c

                          if 0 <= check_row < n and 0 <= check_col < n and self.grid[check_row][check_col]!='X':
                              if self.grid[check_row][check_col] == player:
                                  count_player_cells += 1
                              elif self.grid[check_row][check_col] =='.':
                                  if empty_cell is None:
                                      empty_cell = (check_row, check_col)
                                  # else:
                                  #     # More than one empty cell
                                  #     # empty_cell = None
                                  #     break
                          else:
                            break
                      # Check if  is valid and has a single empty cell
                      if (count_player_cells >= 2 and
                          empty_cell is not None and
                          self.grid[empty_cell[0]][empty_cell[1]] == '.'):

                          if min_moves > 1:
                              min_moves = 4-count_player_cells
                              best_move = empty_cell

                      # If no best move yet, consider potential
                      elif (count_player_cells == 1 and
                            empty_cell is not None and
                            not best_move):
                          min_moves = 3
                          best_move = empty_cell

          return (best_move, min_moves, score) if best_move else (None, -1)
      
################################################################
    def is_game_over(self):
        return all(cell != "." for row in self.grid for cell in row)
    
################################################################
    def best_move (self):
      list_moves = []
      list_moves.append(self.best_diagonal_move())
      list_moves.append(self.check_HV())
      list_moves.append(self.check_square())
      list_moves.append(self.check_best_move_H())
      list_moves.append(self.check_best_move_v())
      # list_moves.append(self.check_L_shape())
      valid_moves = [move for move in list_moves if move[0] is not None]
      if not valid_moves:
        # If no valid moves, return a random move or handle gracefully
        return self.random_play()

    # Sort valid moves based on score and minimum moves
      sorted_data = sorted(valid_moves, key=lambda x: (x[1], -x[2]))
      return sorted_data[0][0][0],sorted_data[0][0][1]
################################################################
    

    def random_play(self):
      while True:
        row = np.random.choice(range(5))
        col = np.random.choice(range(5))
        if self.grid[row][col] != 'X' and self.grid[row][col]!='O':
            return row, col
################################################################
    def play(self):
        print("Welcome to Grid Domination: Pattern Clash with AI!")
        while not self.is_game_over():
            self.print_grid()
            print(f"Current scores: {self.scores}")

            if self.current_player == "P1":
                move = input(f"Enter your move (row col): ").upper()
                col = move[0]  # First character is the column letter
                row = int(move[1]) - 1  # Second character is the row number (adjusted for 0 index)

                # Convert column letter (A, B, C, etc.) to index (0, 1, 2, etc.)
                col_index = ord(col) - ord('A')

                if not self.make_move(row, col_index):
                    print("Invalid move. Try again.")
                    continue
            else:
                print("AI is making its move...")
                if self.start:
                  row, col = self.random_play()
                  self.make_move(row, col)
                  self.start=0
                else:
                  row, col = self.best_move()
                  self.make_move(row, col)
                print(f"AI plays at {row}, {col}")
        self.print_grid()
        winner = self.get_winner()
        print(f"Winner is: {winner}")
################################################################

# Run the game with AI
if __name__ == "__main__":
    game = GridDominationGameAI()
    game.play()

