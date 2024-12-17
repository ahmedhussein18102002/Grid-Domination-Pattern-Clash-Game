# Grid Domination: Pattern Clash

## Overview
**Grid Domination: Pattern Clash** is a strategic grid-based game where players compete to dominate the board by forming patterns and scoring points. The game features an AI opponent and supports a graphical user interface (GUI) for interactive gameplay.

## Features
1. **Pattern-Based Scoring:**
   - Score points by forming specific patterns such as:
     - Horizontal, vertical, or diagonal lines (3-line): **5 points**
     - 2x2 squares: **10 points**
     - Diagonal patterns: **7 points**

2. **Interactive GUI:**
   - A Tkinter-based GUI provides an engaging and easy-to-use interface for gameplay.
   - Players can see their moves and scores updated in real-time.

3. **AI Opponent:**
   - Features a minimax-based AI opponent with alpha-beta pruning for optimal move selection.
   - Dynamic depth adjustment ensures balanced gameplay.

4. **Game Logic:**
   - Turn-based gameplay with alternating moves between the player and the AI.
   - Game ends when a player scores **20 points** or the grid is completely filled.

5. **Customizable Rules:**
   - Scoring and rules are easily adjustable for different game styles.

## Installation
### Prerequisites
- Python 3.x
- Tkinter (usually included with Python)
- Numpy (for efficient computations)

### Steps
1. Clone or download the repository.
2. Ensure all required files are in the same directory:
   - `grid_domination.py`: Contains the game logic and AI implementation.
   - `grid_domination_gui.py`: Implements the graphical interface.
3. Install Numpy if not already installed:
   ```bash
   pip install numpy
   ```

## How to Play
1. Run the GUI application:
   ```bash
   python grid_domination_gui.py
   ```
2. The GUI will display a 5x5 grid.
3. **Player Moves:**
   - Click on an empty cell to place your marker (`X`).
   - Your move is validated in real-time.
4. **AI Moves:**
   - The AI will automatically make its move after yours.
5. **Scoring:**
   - Form patterns to score points:
     - Lines (3-line), squares, or diagonals.
6. **Winning the Game:**
   - The game ends when one player reaches **20 points** or the board is full.

## Game Logic Highlights
1. **Pattern Recognition:**
   - The game detects patterns dynamically for scoring, including 3-line, 2x2 squares, and diagonals.

2. **AI Strategy:**
   - Implements a minimax algorithm with alpha-beta pruning.
   - Considers potential moves and evaluates board states to make the optimal decision.

3. **Grid Management:**
   - A 5x5 grid is used to track moves and patterns.
   - Each cell can hold a player's marker (`X` for Player, `O` for AI) or a special marker for patterns already scored.

## Code Structure
- **`grid_domination.py`:**
  - Core game logic, including:
    - Pattern detection
    - Score updating
    - AI decision-making

- **`grid_domination_gui.py`:**
  - User interface built with Tkinter.
  - Displays the grid, handles user input, and manages game flow.

## Screenshots
1. **Game Start:**
   - A clean 5x5 grid ready for the first move.

2. **Player and AI Moves:**
   - Real-time updates showing player (`X`) and AI (`O`) moves.

3. **Scoreboard:**
   - Displays current scores and game status.

## Customization
- Modify `pattern_scores` in `grid_domination.py` to adjust scoring for different patterns.
- Customize the grid size by changing `grid_dim` and related logic.

## Future Enhancements
- Add multiplayer support.
- Allow custom grid sizes.
- Introduce new scoring patterns.

## Authors
- Developed by Ahmed and collaborators.

## License
This project is open-source and available under the MIT License.

