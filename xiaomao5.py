import random
from typing import Tuple, List, Optional
from gomoku.agents.base import Agent
from gomoku.core.models import GameState, Player
from gomoku.llm.openai_client import OpenAIGomokuClient
#from google.colab import userdata

class StudentLLMAgent(Agent):
    """
    Gomoku AI with pattern scoring + minimax predictive search.
    """

    def __init__(self, agent_id: str, debug: bool = False, search_depth: int = 2):
        super().__init__(agent_id)
        self.debug = debug
        self.search_depth = search_depth
        print(f"🎓 Created StudentLLMAgent: {agent_id} with depth={search_depth}")
        self._pattern_cache = {}
        self.score_map = {
            "live_four": 100000,
            "dead_four": 5000,
            "live_three": 1000,
            "dead_three": 500,
            "live_two": 200
        }
        self._setup()

    def _setup(self) -> None:
        self.system_prompt = self._create_system_prompt()
        self.llm_client = OpenAIGomokuClient(
            #api_key=userdata.get('PROF_API_KEY'),
            model="qwen/qwen3-8b",
            #endpoint="https://api.mtkachenko.info/v1",
        )
        print("✅ Agent setup complete!")

    def _create_system_prompt(self) -> str:
        return """
GOMOKU MASTER STRATEGY:
1. IMMEDIATE WIN
2. BLOCK WIN
3. FORK CREATION
4. POSITIONAL ADVANTAGE
""".strip()

    # === MOVE EVALUATION ===
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

        # Positional bonus
        rows, cols = len(board), len(board[0])
        center_row, center_col = rows // 2, cols // 2
        dist = abs(row - center_row) + abs(col - center_col)
        score += max(0, 50 - dist * 5)

        self._pattern_cache[cache_key] = score
        return score

    def _board_score(self, board: List[List[str]], player: Player) -> int:
        """Evaluates the entire board for the player using pattern scores."""
        legal_moves = [(r, c) for r in range(len(board)) for c in range(len(board[0])) if board[r][c] == '.']
        return sum(self._evaluate_move(board, move, player) for move in legal_moves)

    # === MINIMAX WITH ALPHA-BETA PRUNING ===
    def _minimax(self, board: List[List[str]], depth: int, alpha: int, beta: int, player: Player, maximizing: bool) -> int:
        opponent = Player('X') if player == Player('O') else Player('O')

        # Terminal check
        if depth == 0:
            return self._board_score(board, player if maximizing else opponent)

        legal_moves = [(r, c) for r in range(len(board)) for c in range(len(board[0])) if board[r][c] == '.']
        if not legal_moves:
            return 0

        if maximizing:
            max_eval = -float('inf')
            for move in legal_moves:
                if self._is_winning_move(board, move, player):
                    return float('inf')  # Immediate win

                r, c = move
                board[r][c] = player.value
                eval = self._minimax(board, depth - 1, alpha, beta, player, False)
                board[r][c] = '.'
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                if self._is_winning_move(board, move, opponent):
                    return -float('inf')  # Opponent immediate win

                r, c = move
                board[r][c] = opponent.value
                eval = self._minimax(board, depth - 1, alpha, beta, player, True)
                board[r][c] = '.'
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    # === GAMEPLAY DECISION ===
    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        self._pattern_cache.clear()
        board = game_state.board
        current_player = game_state.current_player
        opponent = Player('X') if current_player == Player('O') else Player('O')
        legal_moves = game_state.get_legal_moves()

        if not legal_moves:
            raise RuntimeError("No legal moves available.")

        # Priority 1: Win now
        for move in legal_moves:
            if self._is_winning_move(board, move, current_player):
                return move

        # Priority 2: Block opponent win
        for move in legal_moves:
            if self._is_winning_move(board, move, opponent):
                return move

        # Priority 3: Fork creation
        fork_move = self._find_fork_move(board, legal_moves, current_player)
        if fork_move:
            return fork_move

        # Minimax decision
        best_score = -float('inf')
        best_move = None

        for move in legal_moves:
            r, c = move
            board[r][c] = current_player.value
            score = self._minimax(board, self.search_depth - 1, -float('inf'), float('inf'), current_player, False)
            board[r][c] = '.'

            if score > best_score or (score == best_score and self._evaluate_move(board, move, current_player) > self._evaluate_move(board, best_move, current_player)):
                best_score = score
                best_move = move

            if self.debug:
                print(f"Move {move} → Predicted Score: {score}")

        return best_move if best_move else random.choice(legal_moves)

    # === EXISTING HELPERS (forks, patterns, etc.) ===
    def _find_fork_move(self, board, legal_moves, player):
        opponent = Player('X') if player == Player('O') else Player('O')
        for move in legal_moves:
            threats = self._count_threats(board, move, player)
            if threats >= 2:
                r, c = move
                board[r][c] = player.value
                opp_moves = [(rr, cc) for rr in range(len(board))
                             for cc in range(len(board[0])) if board[rr][cc] == '.']
                if not any(self._is_winning_move(board, omv, opponent) for omv in opp_moves):
                    board[r][c] = '.'
                    return move
                board[r][c] = '.'
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
            # forward
            r, c = row + dr, col + dc
            while 0 <= r < rows and 0 <= c < cols and board[r][c] == player.value:
                count += 1
                r += dr
                c += dc
            # backward
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

    # === Pattern Match Methods ===
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
