# Import the necessary modules
import random
from typing import Tuple

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
        """Create a strategic and precise Gomoku guidance prompt."""
        return """
    Gomoku Strategic Mastery Framework:

    CORE STRATEGIC OBJECTIVES:
    1. Immediate Victory Detection
    - Scan entire board for 5-piece winning configurations
    - Execute winning move without hesitation if available

    2. Threat Neutralization
    - Proactively identify and block opponent's potential winning sequences
    - Prevent opponent from establishing critical board control

    3. Strategic Board Control
    - Prioritise central board positioning
    - Develop multi-directional attack potential
    - Create complex threat scenarios with fork opportunities

    DECISION-MAKING PROTOCOL:
    - Validate move against strict positional constraints
    - Evaluate move across multiple strategic dimensions
    - Balance offensive advancement with defensive resilience

    RESPONSE REQUIREMENTS:
    {
        "reasoning": "Precise strategic rationale",
        "row": <validated_row_index>,
        "col": <validated_column_index>
    }

    CRITICAL CONSTRAINTS:
    - Select only empty board positions ('.')
    - Ensure move coordinates are valid and strategically sound
    - Provide clear, concise strategic justification
    """.strip()


    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        print(f"\n🧠 {self.agent_id} is thinking...")

        try:
            # Create messages with an ultra-concise prompt
            board_str = game_state.format_board(formatter="standard")
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""Current board state:
            {board_str}

            Game Context:
            - Current Player: {game_state.current_player.value}
            - Total Moves: {len(game_state.move_history) if hasattr(game_state, 'move_history') else 'Unknown'}
            - Board Dimensions: 8x8

            STRATEGIC ANALYSIS REQUEST:
            Carefully evaluate the current board state and strategically select your next move. 

            Key Evaluation Criteria:
            1. Immediate win potential
            2. Opponent threat blocking
            3. Board control opportunities
            4. Multi-directional attack possibilities

            Output your selection strictly following the JSON response format previously specified.
            """}
            ]

            # Set extremely short timeout for LLM
            response = await asyncio.wait_for(
                self.llm_client.complete(messages), 
                timeout=19.9  # Ultra-short 1900ms timeout
            )

            if m := re.search(r"{[^}]+}", response, re.DOTALL):
                json_data = json.loads(m.group(0).strip())
                return json_data["row"], json_data["col"]

        except Exception as e:
            print(e)

        return self._get_fallback_move(game_state)

    def _get_fallback_move(self, game_state: GameState) -> Tuple[int, int]:
        """Simple fallback when LLM fails."""
        return random.choice(game_state.get_legal_moves())
