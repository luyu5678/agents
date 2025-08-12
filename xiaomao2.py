# Import the necessary modules
import random
from typing import List,Tuple

# Import the game framework components
from gomoku.agents.base import Agent
from gomoku.core.models import GameState, Player
from gomoku.arena import GomokuArena
from gomoku.utils import ColorBoardFormatter
from gomoku.llm.openai_client import OpenAIGomokuClient
# from google.colab import userdata
import re
import json

# First, let's design our LLM agent class structure

class StudentLLMAgent(Agent):
    """An educational LLM agent that students will build step by step."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        print(f"🎓 Created StudentLLMAgent: {agent_id}")

    def _setup(self):
        """Setup our LLM client and prompts."""
        print("⚙️  Setting up LLM agent...")

        # We'll add the LLM client setup here
        # For now, let's define our system prompt
        self.system_prompt = self._create_system_prompt()

        # We'll simulate an LLM for educational purposes
        self.llm_client = OpenAIGomokuClient(
            # api_key=userdata.get('PROF_API_KEY'),
            model="qwen/qwen3-8b",
            # endpoint="https://api.mtkachenko.info/v1",
        )

        print("✅ Agent setup complete!")
    def _create_system_prompt(self) -> str:
        """Create an advanced Gomoku strategic guidance prompt."""
        return """
    GOMOKU MASTER STRATEGY FRAMEWORK v2.0

    ABSOLUTE PRIORITY SEQUENCE:
    1. IMMEDIATE VICTORY DETECTION
    - 100% PRIORITY: Scan for 5-in-a-row winning configurations
    - EXECUTE WINNING MOVE WITHOUT HESITATION
    - ZERO TOLERANCE FOR MISSED WINNING OPPORTUNITIES

    2. THREAT NEUTRALIZATION PROTOCOL
    - Proactively block opponent's potential winning sequences
    - Detect and neutralize 4-piece threat sequences
    - Prevent opponent's fork opportunities

    3. STRATEGIC BOARD DOMINATION
    - CENTRAL BOARD CONTROL IS CRITICAL
        * Prioritize moves near (4,4) as primary strategic point
        * Develop multi-directional attack vectors
        * Create complex threat scenarios

    4. ADVANCED POSITIONING STRATEGY
    - Analyze board for:
        * Potential fork creation
        * Blocking opponent's strategic lines
        * Maximizing move flexibility

    DECISION OPTIMIZATION RULES:
    - Validate ONLY empty board positions ('.')
    - Evaluate moves across 4 strategic dimensions:
    1. Immediate win potential
    2. Threat blocking effectiveness
    3. Board control magnitude
    4. Future move flexibility

    RESPONSE MANDATORY FORMAT:
    {
        "reasoning": "Precise strategic rationale",
        "row": <optimal_row_index>,
        "col": <optimal_column_index>
    }

    CRITICAL CONSTRAINTS:
    - SPEED is essential
    - STRATEGIC DEPTH is non-negotiable
    - ZERO invalid moves allowed
    """.strip()

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        print(f"\n🧠 {self.agent_id} is thinking...")

        try:
            # Rapid board state analysis
            board_str = game_state.format_board(formatter="standard")
            current_player = game_state.current_player.value
            legal_moves = game_state.get_legal_moves()

            # Prioritized move selection
            strategic_moves = self._analyze_strategic_moves(board_str, current_player, legal_moves)
            
            if strategic_moves:
                return strategic_moves[0]

            # Fallback to comprehensive LLM strategy
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""BOARD STATE:
    {board_str}

    STRATEGIC CONTEXT:
    - Player: {current_player}
    - Legal Moves: {len(legal_moves)}
    - Board Control Analysis Required

    CRITICAL MISSION:
    Perform an ultra-precise move selection that:
    1. Maximizes winning probability
    2. Minimizes opponent's strategic options
    3. Establishes board dominance

    OUTPUT MANDATORY: 
    - Fastest strategic decision
    - JSON format compliance
    - Zero hesitation
    """}
            ]

            # Implement ultra-fast LLM response mechanism
            response = await asyncio.wait_for(
                self.llm_client.complete(messages), 
                timeout=1.0  # Reasonable timeout
            )

            # Robust parsing mechanism
            if m := re.search(r'{\s*"row"\s*:\s*(\d+)\s*,\s*"col"\s*:\s*(\d+)\s*}', response):
                return int(m.group(1)), int(m.group(2))

        except Exception as e:
            print(f"Strategic decision error: {e}")

        # Ultimate fallback strategy
        return self._get_fallback_move(game_state)

    def _analyze_strategic_moves(self, board_str: str, current_player: str, legal_moves: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Perform rapid strategic move analysis
        Prioritization hierarchy:
        1. Immediate winning moves
        2. Threat blocking moves
        3. Central board control
        4. Strategic positioning
        """
        strategic_priority_moves = []

        # Immediate win detection
        for move in legal_moves:
            if self._is_winning_move(board_str, move, current_player):
                strategic_priority_moves.append(move)

        # Central board prioritization
        central_moves = [
            (4, 4),   # Absolute center
            (3, 4), (5, 4),  # Vertical center line
            (4, 3), (4, 5),  # Horizontal center line
            (3, 3), (3, 5), (5, 3), (5, 5)  # Diagonal center areas
        ]
        
        strategic_central_moves = [
            move for move in central_moves if move in legal_moves
        ]
        
        strategic_priority_moves.extend(strategic_central_moves)

        return strategic_priority_moves

    def _is_winning_move(self, board_str: str, move: Tuple[int, int], player: str) -> bool:
        """Rapid winning move detection"""
        # Implement efficient winning move detection logic
        # This is a placeholder - replace with actual implementation
        return False

    def _get_fallback_move(self, game_state: GameState) -> Tuple[int, int]:
        """Ultimate fallback strategy with central board preference"""
        legal_moves = game_state.get_legal_moves()
        
        # Prioritize center and center-adjacent moves
        center_moves = [
            (4, 4), (4, 3), (4, 5), 
            (3, 4), (5, 4)
        ]
        
        for move in center_moves:
            if move in legal_moves:
                return move
        
        return random.choice(legal_moves)
