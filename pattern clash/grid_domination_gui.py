
import tkinter as tk
import tkinter.messagebox as messagebox
import numpy as np

class GridDominationGUI:
    def __init__(self, game):
        """
        Initialize the GUI for the Grid Domination game.
        
        Args:
            game (GridDominationGameAI): The game logic instance
        """
        self.game = game
        
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Grid Domination: Pattern Clash")
        
        # Create a frame for the game board
        self.board_frame = tk.Frame(self.root, bg='white')
        self.board_frame.pack(padx=20, pady=20)
        
        # Create buttons for the grid
        self.buttons = []
        for row in range(5):
            row_buttons = []
            for col in range(5):
                # Create button for each grid cell
                btn = tk.Button(
                    self.board_frame, 
                    text=' ', 
                    width=4, 
                    height=2, 
                    command=lambda r=row, c=col: self.on_cell_click(r, c),
                    font=('Arial', 12, 'bold')
                )
                btn.grid(row=row, column=col, padx=2, pady=2)
                row_buttons.append(btn)
            self.buttons.append(row_buttons)
        
        # Create a frame for game information
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(pady=10)
        
        # Labels for scores and current player
        self.score_label = tk.Label(
            self.info_frame, 
            text=f"Player Score: 0 | AI Score: 0", 
            font=('Arial', 12)
        )
        self.score_label.pack()
        
        self.player_label = tk.Label(
            self.info_frame, 
            text="Your Turn", 
            font=('Arial', 12, 'bold')
        )
        self.player_label.pack()
        
    def on_cell_click(self, row, col):
        """
        Handle cell click events for player moves.
        
        Args:
            row (int): Row of the clicked cell
            col (int): Column of the clicked cell
        """
        
        # Attempt to make a move
        if self.game.make_move(row, col):
            # Update the button appearance
            # Player uses 'X' (blue)
            self.buttons[row][col].config(
                text='X', 
                state=tk.DISABLED, 
                disabledforeground='blue'
            )
            # Update scores
            self.score_label.config(
                text=f"Player Score: {self.game.scores['P1']} | AI Score: {self.game.scores['P2']}"
            )
            self.hashing()
            # Check if game is over
            if self.game.is_game_over():
                self.end_game()
            else:
                # AI's turn
                self.player_label.config(text="AI's Turn")
                self.root.update()
                
                # AI move
                if self.game.start:
                    ai_row, ai_col = self.game.random_play()
                    self.game.start = 0
                else:
                    ai_row, ai_col = self.game.best_move()
                
                self.game.make_move(ai_row, ai_col)
                
                # Update button for AI's move (AI uses 'O' in red)
                self.buttons[ai_row][ai_col].config(
                    text='O', 
                    state=tk.DISABLED, 
                    disabledforeground='red'
                )
                
                # Update scores again
                self.score_label.config(
                    text=f"Player Score: {self.game.scores['P1']} | AI Score: {self.game.scores['P2']}"
                )
                self.hashing()
                # Check if game is over after AI move
                if self.game.is_game_over():
                    self.end_game()
                else:
                    # Back to player's turn
                    self.player_label.config(text="Your Turn")
            
        else:
            # Invalid move
            messagebox.showwarning("Invalid Move", "This cell is already occupied!")
    
    def hashing(self):
        for r,c in self.game.hash:
          self.buttons[r][c].config(
              text='#', 
              state=tk.DISABLED, 
              disabledforeground='black'
          )
        self.game.hash=[]
    
    def end_game(self):
        """
        Handle end of game, show winner and final scores.
        """
        winner = self.game.get_winner()
        # Map game winners to more readable format
        winner_map = {
            'P1': 'Player',
            'P2': 'AI',
            'Tie': 'No one (It\'s a Tie!)'
        }
        messagebox.showinfo(
            "Game Over", 
            f"Game Ended!\n"
            f"Winner: {winner_map[winner]}\n"
            f"Final Scores:\n"
            f"Player: {self.game.scores['P1']}\n"
            f"AI: {self.game.scores['P2']}"
        )
        self.game.grid = [["." for _ in range(5)] for _ in range(5)]
        self.root.quit()
        self.root.destroy()
        
    
    def run(self):
        """
        Start the GUI game loop.
        """
        self.root.mainloop()

def main():
    from grid_domination import GridDominationGameAI  # Import your game class
    
    # Create game instance
    game = GridDominationGameAI()
    
    # Create and run GUI
    gui = GridDominationGUI(game)
    gui.run()

if __name__ == "__main__":
    main()
