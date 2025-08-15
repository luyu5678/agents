import random
import copy
from typing import Tuple, List, Optional, Dict
from gomoku.agents.base import Agent
from gomoku.core.models import GameState, Player
from gomoku.llm.openai_client import OpenAIGomokuClient
from concurrent.futures import ThreadPoolExecutor
#from google.colab import userdata

class StudentLLMAgent(Agent):
    """
    Gomoku AI with adaptive depth, move ordering, candidate restriction, caching, and parallel minimax.
    """

    def __init__(self, agent_id: str, debug: bool = False, base_depth: int = 2):
        super().__init__(agent_id)
        self.debug = debug
        self.base_depth = base_depth
        self._pattern_cache: Dict = {}
        self._transposition_table: Dict = {}
        self.score_map = {
        "live_four": 150000,
        "dead_four": 8000,
        "live_three": 5000,
        "dead_three": 2000,
        "live_two": 800
    }

        self._setup()

    def _setup(self) -> None:
        self.system_prompt = self._create_system_prompt()
        self.llm_client = OpenAIGomokuClient(
            #api_key=userdata.get('PROF_API_KEY'),
            model="qwen/qwen3-8b",
            #endpoint="https://api.mtkachenko.info/v1",
        )

    def _create_system_prompt(self) -> str:
        return """
GOMOKU MASTER STRATEGY:
1. IMMEDIATE WIN
2. BLOCK WIN
3. FORK CREATION
4. POSITIONAL ADVANTAGE
""".strip()

    def _evaluate_move(self, board: List[List[str]], move: Tuple[int, int], player: Player) -> int:
        cache_key = (tuple(map(tuple, board)), move, player)
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]
        symbol = player.value
        row, col = move
        score = 0
        for pattern_name, pattern_score in self.score_map.items():
            matcher_method = getattr(self, f"_match_{pattern_name}")
            score += matcher_method(board, row, col, symbol) * pattern_score
        rows, cols = len(board), len(board[0])
        center_row, center_col = rows // 2, cols // 2
        dist = abs(row - center_row) + abs(col - center_col)
        score += max(0, 50 - dist * 5)
        self._pattern_cache[cache_key] = score
        return score

    def _board_score(self, board: List[List[str]], player: Player) -> int:
        legal_moves = self._candidate_moves(board)
        return sum(self._evaluate_move(board, move, player) for move in legal_moves)

    def _candidate_moves(self, board: List[List[str]]) -> List[Tuple[int, int]]:
        rows, cols = len(board), len(board[0])
        candidates = set()
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for r in range(rows):
            for c in range(cols):
                if board[r][c] != '.':
                    for dr, dc in directions:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == '.':
                            candidates.add((nr, nc))
        moves_list = list(candidates) if candidates else [(r, c) for r in range(rows) for c in range(cols) if board[r][c] == '.']
        center_row, center_col = rows // 2, cols // 2
        return sorted(moves_list, key=lambda m: (
            -self._evaluate_move(board, m, Player('O')),  # adjust for current player
            abs(m[0]-center_row) + abs(m[1]-center_col)
        ))


    def _minimax(self, board, depth, alpha, beta, player, maximizing):
        key = (tuple(map(tuple, board)), player, depth, maximizing)
        if key in self._transposition_table:
            return self._transposition_table[key]
        opponent = Player('X') if player == Player('O') else Player('O')
        if depth == 0:
            score = self._board_score(board, player if maximizing else opponent)
            self._transposition_table[key] = score
            return score
        legal_moves = sorted(self._candidate_moves(board), key=lambda m: -self._evaluate_move(board, m, player))
        if not legal_moves:
            return 0
        
        # Early termination checks
        if maximizing and any(self._is_winning_move(board, m, player) for m in legal_moves):
            return float('inf')
        if not maximizing and any(self._is_winning_move(board, m, opponent) for m in legal_moves):
            return -float('inf')
        
        if maximizing:
            value = -float('inf')
            for move in legal_moves:
                if self._is_winning_move(board, move, player):
                    return float('inf')
                r, c = move
                board[r][c] = player.value
                value = max(value, self._minimax(board, depth - 1, alpha, beta, player, False))
                board[r][c] = '.'
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
        else:
            value = float('inf')
            for move in legal_moves:
                if self._is_winning_move(board, move, opponent):
                    return -float('inf')
                r, c = move
                board[r][c] = opponent.value
                value = min(value, self._minimax(board, depth - 1, alpha, beta, player, True))
                board[r][c] = '.'
                beta = min(beta, value)
                if beta <= alpha:
                    break
        self._transposition_table[key] = value
        return value

    def _adaptive_depth(self, num_moves: int) -> int:
        if num_moves <= 8:
            return self.base_depth + 2
        elif num_moves <= 25:
            return self.base_depth + 3
        elif num_moves >= 40:
            return max(1, self.base_depth - 1)
        return self.base_depth

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        self._pattern_cache.clear()
        self._transposition_table.clear()
        board = game_state.board
        current_player = game_state.current_player
        opponent = Player('X') if current_player == Player('O') else Player('O')
        legal_moves = self._candidate_moves(board)
        if not legal_moves:
            raise RuntimeError("No legal moves available.")
        for move in legal_moves:
            if self._is_winning_move(board, move, current_player):
                return move
        for move in legal_moves:
            if self._is_winning_move(board, move, opponent):
                return move
        fork_move = self._find_fork_move(board, legal_moves, current_player)
        if fork_move:
            return fork_move
        depth = self._adaptive_depth(len(legal_moves))
        best_score = -float('inf')
        best_move = None
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self._score_move, move, board, current_player, depth): move for move in legal_moves}
            for future in futures:
                move = futures[future]
                score = future.result()
                if best_move is None or score > best_score or \
                   (score == best_score and self._evaluate_move(board, move, current_player) > self._evaluate_move(board, best_move, current_player)):
                    best_score = score
                    best_move = move
                if self.debug:
                    print(f"Move {move} → Predicted Score: {score}")
        return best_move if best_move else random.choice(legal_moves)

    def _score_move(self, move, board, player, depth):
        temp_board = copy.deepcopy(board)  # Thread-safe
        r, c = move
        temp_board[r][c] = player.value
        score = self._minimax(temp_board, depth - 1, -float('inf'), float('inf'), player, False)
        return score

    def _find_fork_move(self, board, legal_moves, player):
        opponent = Player('X') if player == Player('O') else Player('O')
        # First, create own fork
        for move in legal_moves:
            if self._count_threats(board, move, player) >= 2:
                r, c = move
                board[r][c] = player.value
                if not any(self._is_winning_move(board, omv, opponent) for omv in self._candidate_moves(board)):
                    board[r][c] = '.'
                    return move
                board[r][c] = '.'
        # Then, block opponent fork
        for move in legal_moves:
            if self._count_threats(board, move, opponent) >= 2:
                return move
        return None


    def _is_winning_move(self, board, move, player):
        row, col = move
        if board[row][col] != '.':
            return False
        board[row][col] = player.value
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        rows, cols = len(board), len(board[0])
        for dr, dc in directions:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < rows and 0 <= c < cols and board[r][c] == player.value:
                count += 1
                r += dr
                c += dc
            r, c = row - dr, col - dc
            while 0 <= r < rows and 0 <= c < cols and board[r][c] == player.value:
                count += 1
                r -= dr
                c -= dc
            if count >= 5:
                board[row][col] = '.'
                return True
        board[row][col] = '.'
        return False

    def _count_threats(self, board, move, player):
        symbol = player.value
        r, c = move
        patterns = [self._match_live_four, self._match_dead_four, self._match_live_three]
        return sum(m(board, r, c, symbol) for m in patterns)

    def _match_live_four(self, board, r, c, symbol):
        return self._count_pattern(board, r, c, symbol, 4, 2)

    def _match_dead_four(self, board, r, c, symbol):
        return self._count_pattern(board, r, c, symbol, 4, 1)

    def _match_live_three(self, board, r, c, symbol):
        return self._count_pattern(board, r, c, symbol, 3, 2)

    def _match_dead_three(self, board, r, c, symbol):
        return self._count_pattern(board, r, c, symbol, 3, 1)

    def _match_live_two(self, board, r, c, symbol):
        return self._count_pattern(board, r, c, symbol, 2, 2)

    def _count_pattern(self, board, r, c, symbol, target_count, open_ends):
        if board[r][c] != '.':
            return 0
        rows, cols = len(board), len(board[0])
        board[r][c] = symbol
        matches = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            rr, cc = r + dr, c + dc
            while 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == symbol:
                count += 1
                rr += dr
                cc += dc
            forward_empty = (0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == '.')
            rr, cc = r - dr, c - dc
            while 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == symbol:
                count += 1
                rr -= dr
                cc -= dc
            backward_empty = (0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == '.')
            if count == target_count and (int(forward_empty) + int(backward_empty)) >= open_ends:
                matches += 1
        board[r][c] = '.'
        return matches
