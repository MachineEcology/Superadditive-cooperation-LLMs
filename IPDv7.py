import os
from typing import Dict, List, Tuple, Literal, Any, Optional, Annotated
import random
from itertools import combinations, product
import json
import re

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from nicerthanhumansmetrics import evaluate_player

# Set LangSmith environment variables
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "IPD-tournament-simulation"

# Configuration
PAYOFFS = {
    ("action_a", "action_a"): (3, 3),  # Cooperation (C,C)
    ("action_a", "action_b"): (-1, 5),  # Defection (C,D)
    ("action_b", "action_a"): (5, -1),  # Defection (D,C)
    ("action_b", "action_b"): (0, 0)  # Mutual defection (D,D)
}

NUM_GROUPS = 3  # Number of groups
GROUP_SIZE = 2  # Players per group
MAX_ROUNDS_PER_MATCH = 20
MAX_TOTAL_ROUNDS = 100
INTERGROUP_COMPETITION = True  # Enable intergroup competition
GROUP_REWARD_MULTIPLIER = 2.0  # Bonus for the winning group
SIMPLIFY_MATCHES_GRAPH = False  # Set to true if you want to reduce the number of matches
MAX_MATCHES_NUM = 30  # Max of matches played, only works if SIMPLIFY_MATCHES_GRAPH is True
CRITIQUE_ITERATIONS = 1
PLANNING_FREQUENCY = 5  # with 0 there is no planning
COOPERATE_DEFECT_PROMPT = False
LLM = OllamaLLM(model="qwen3:14b")

# 2-vs-1 Game Configuration
TEAM_GAME_MODE = True  # Toggle between 1v1 and 2v1 modes
DISCUSSION_TURNS = 3           # Number of pre-game strategizing turns for Team 1
ADVERSARY_DISCUSSION_TURNS = 3 # Number of pre-game strategizing turns for Team 2 (adversary)
TEAM1_SIZE = 2         # Number of agents in Team 1
TEAM2_SIZE = 1         # Number of agents in Team 2

# Payoff Matrix for 2-vs-1 Game
# Format: (team1_agent1_move, team1_agent2_move, team2_move) -> (team1_total_score, team2_score)
# TODO: Fill in payoff values
PAYOFFS_2V1 = {
    ("action_a", "action_a", "action_a"): (0, 0),    # All cooperate - FILL IN
    ("action_a", "action_a", "action_b"): (0, 0),    # Team1 both cooperate, Team2 defects - FILL IN
    ("action_a", "action_b", "action_a"): (0, 0),    # Team1 mixed, Team2 cooperates - FILL IN
    ("action_a", "action_b", "action_b"): (0, 0),    # Team1 mixed, Team2 defects - FILL IN
    ("action_b", "action_a", "action_a"): (0, 0),    # Team1 mixed (reversed), Team2 cooperates - FILL IN
    ("action_b", "action_a", "action_b"): (0, 0),    # Team1 mixed (reversed), Team2 defects - FILL IN
    ("action_b", "action_b", "action_a"): (0, 0),    # Team1 both defect, Team2 cooperates - FILL IN
    ("action_b", "action_b", "action_b"): (0, 0),    # All defect - FILL IN
}


def merge_dict(current: dict, update: dict) -> dict:
    new_dict = current.copy()
    new_dict.update(update)
    return new_dict


def custom_json_encoder(obj):
    """Handle special types during JSON serialization."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    elif isinstance(obj, Tuple):
        return list(obj)
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    # Add any other special type handling as needed
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# %% Classes


class Action(BaseModel):
    move: Literal["action_a", "action_b"]
    reasoning: str
    keep_playing: bool


class MetaPromptFields(BaseModel):
    min_max: Tuple[int, int] = Field(description="lowest and highest payoff a player can get in a single round")
    actions: List[str] = Field(description="actions is player A allowed to play")
    payoff: int = Field(description="player X’s payoff in a single round if X plays p and Y plays q")
    round: int = Field(description="current round of the game")
    action: Literal["action_a", "action_b"] = Field(description="action player X played in round i")
    points: int = Field(description="points player X collected in round i")
    num_actions: int = Field(description="how many times did player X choose action p")
    num_points: int = Field(description="what is player's X current total score")
    tft: bool = Field(
        description="whether player X uses a Tit For Tat strategy (will first cooperate, then subsequently replicate an opponent's previous action)")
    forgiving: bool = Field(
        description="whether player X is forgiving (Propensity to choose action_a again after an opponent’s action_b)")


class PlayerStats(BaseModel):
    total_score: int = Field(default=0)
    action_a_count: int = Field(default=0)
    action_b_count: int = Field(default=0)
    move_history: List[str] = Field(default_factory=list)
    score_history: List[int] = Field(default_factory=list)
    action_a_rate_history: List[float] = Field(default_factory=list)
    group_id: int = Field(default=0)
    plan: str = Field(default="No plan yet")
    critique: str = Field(default="No plan yet")
    SFEM: List[Dict[str, Any]] = Field(default_factory=list)
    traits: List[Dict[str, Any]] = Field(default_factory=list)
    meta_prompt_results: List[Dict[str, Any]] = Field(default_factory=list)

    # Team-specific fields
    is_team_member: bool = Field(default=False)
    team_id: Optional[int] = Field(default=None)  # Which team (1 or 2)
    teammate_ids: List[int] = Field(default_factory=list)  # For Team 1 members

    class Config:
        arbitrary_types_allowed = True


class GroupStats(BaseModel):
    group_id: int
    total_score: int = Field(default=0)
    members: List[int] = Field(default_factory=list)
    avg_cooperation_rate: float = Field(default=0.0)
    won_competitions: int = Field(default=0)


class MatchState(BaseModel):
    # Team configuration
    is_team_game: bool = Field(default=False)
    team1_member_ids: List[int] = Field(default_factory=list)  # [agent1_id, agent2_id]
    team2_member_id: int = Field(default=0)

    # Legacy 1v1 fields (keep for backward compatibility)
    player1_id: int = Field(default=0)
    player2_id: int = Field(default=0)

    # Current round tracking
    current_round: int = Field(default=1)

    # Stats tracking
    # For team games: team1_stats maps member_id -> PlayerStats
    team1_stats: Dict[int, PlayerStats] = Field(default_factory=dict)
    team2_stats: PlayerStats = Field(default_factory=PlayerStats)

    # Legacy 1v1 stats (keep for backward compatibility)
    player1_stats: PlayerStats = Field(default_factory=PlayerStats)
    player2_stats: PlayerStats = Field(default_factory=PlayerStats)

    # Moves per round
    team1_moves: Dict[int, str] = Field(default_factory=dict)  # {member_id: move}
    team2_move: str = Field(default="")

    # Legacy 1v1 moves
    player1_move: str = Field(default="")
    player2_move: str = Field(default="")

    # Keep playing flags
    team1_keep_playing: Dict[int, bool] = Field(default_factory=dict)
    team2_keep_playing: bool = Field(default=True)

    # Legacy flags
    player1_keep_playing: bool = Field(default=True)
    player2_keep_playing: bool = Field(default=True)

    # Results tracking
    completed: bool = Field(default=False)
    round_results: List[Any] = Field(default_factory=list)  # Team: dict, 1v1: Tuple[str, str]
    round_scores: List[Any] = Field(default_factory=list)   # Team: Tuple[int, int], 1v1: Tuple[int, int]
    is_first_interaction: bool = Field(default=True)

    # Team coordination
    discussion_history: List[Dict[str, Any]] = Field(default_factory=list)  # [{agent_id: int, message: str}]
    team1_shared_plan: str = Field(default="No shared plan yet")


class TournamentState(BaseModel):
    # Track player stats throughout the games
    players: Dict[int, PlayerStats] = Field(default_factory=dict)
    matches: Annotated[dict[int, MatchState], merge_dict] = Field(default_factory=dict)
    current_match_idx: int = Field(default=0)
    groups: Dict[int, GroupStats] = Field(default_factory=dict)  # Track group stats
    round_number: int = Field(default=0)  # Track global round number
    intergroup_competition_results: List[Dict] = Field(default_factory=list)  # keeps track of history of group stats
    first_interaction_coop_intragroup: List[bool] = Field(default_factory=list)
    first_interaction_coop_intergroup: List[bool] = Field(default_factory=list)
    experiment_condition: str = Field(default="super_additive")

    def update_player_stats(self, pid, match_player_stats, match_opponent_stats):
        # Update scores and action counts
        self.players[pid].total_score += match_player_stats.total_score
        self.players[pid].action_a_count += match_player_stats.action_a_count
        self.players[pid].action_b_count += match_player_stats.action_b_count

        # Update histories
        self.players[pid].move_history.extend(match_player_stats.move_history)
        self.players[pid].score_history.extend(match_player_stats.score_history)
        self.players[pid].action_a_rate_history += match_player_stats.action_a_rate_history

        # Update plan
        self.players[pid].plan = match_player_stats.plan
        self.players[pid].critique = match_player_stats.critique

    def update_nth_stats(self, pid, match_player_stats, match_opponent_stats):
        # Update SFEM and traits
        affinities, traits = evaluate_player(match_player_stats.move_history, match_opponent_stats.move_history,
                                             "action_a")
        self.players[pid].SFEM.append(affinities)
        self.players[pid].traits.append(traits)

    def update_group_stats(self) -> Dict[int, GroupStats]:
        """Update statistics for all groups and apply competition rewards if enabled"""
        # Reset group scores for this round
        for group_id in self.groups:
            self.groups[group_id].total_score = 0
            cooperation_counts = []

            # Sum up scores from all group members
            for player_id in self.groups[group_id].members:
                player = self.players[player_id]
                self.groups[group_id].total_score += player.total_score

                total_moves = player.action_a_count + player.action_b_count
                if total_moves > 0:
                    cooperation_counts.append(player.action_a_count / total_moves)

            # Calculate average cooperation rate for the group
            self.groups[group_id].avg_cooperation_rate = sum(cooperation_counts) / len(
                cooperation_counts) if cooperation_counts else 0

        # Apply group bonus if intergroup competition is enabled
        if INTERGROUP_COMPETITION:
            ranked_groups = sorted(self.groups.items(), key=lambda x: x[1].total_score, reverse=True)
            winning_group_id = ranked_groups[0][0]

            if self.current_match_idx == len(self.matches) - 1:
                for player_id in self.groups[winning_group_id].members:
                    self.players[player_id].total_score = int(
                        self.players[player_id].total_score * GROUP_REWARD_MULTIPLIER)

            self.groups[winning_group_id].won_competitions += 1

            # Record competition results
            self.intergroup_competition_results.append({
                "round": self.round_number,
                "winning_group": winning_group_id,
                "group_scores": {g_id: g_stats.total_score for g_id, g_stats in self.groups.items()},
                "cooperation_rates": {g_id: g_stats.avg_cooperation_rate for g_id, g_stats in self.groups.items()}
            })

        return self.groups

    def record_first_interaction(self, player1_id: int, player2_id: int, p1_move: str, p2_move: str):
        """Record cooperation in first interaction between two players"""
        # Check if players are in same group
        same_group = self.players[player1_id].group_id == self.players[player2_id].group_id

        # Consider cooperation as choosing action_a
        p1_cooperated = p1_move == "action_a"
        p2_cooperated = p2_move == "action_a"

        # Record cooperation rates for the appropriate group type
        if same_group:
            self.first_interaction_coop_intragroup.append(p1_cooperated)
            self.first_interaction_coop_intragroup.append(p2_cooperated)
        else:
            self.first_interaction_coop_intergroup.append(p1_cooperated)
            self.first_interaction_coop_intergroup.append(p2_cooperated)


# %% Prompts and format functions

meta_prompt = ChatPromptTemplate.from_template("""
You are Player {player_id} in a strategic game tournament, you are playing against player {opponent_id}.

Game Rules:
{game_rules}

{group_stats}

Round-by-round results:
{round_results}

Reply to some questions about the game rules, progression, state, and opponent strategies:
What is the lowest and highest payoff a player can get in a single round?
Which actions is a player allowed to play?
Which is player 0’s payoff in a single round if player 0 plays {random_action_1} and player 1 plays {random_action_2}?
Which is the current round of the match?
Which action did the opponent play in round {random_round_1} of the latest match?
How many points did the opponent collect in round {random_round_2} of the latest match?
How many times did the opponent choose action {random_action_1} in the latest match?
What is the opponent's total score for the current match?
Does the opponent follow a Tit For Tat strategy? (will first cooperate, then subsequently replicate an opponent's previous action)
Is the opponent forgiving? (Propensity to choose action_a again after an opponent’s action_b) 

{format_instructions}
""")

player_prompt = ChatPromptTemplate.from_template("""
You are Player {player_id} in a strategic game tournament, you are playing against player {opponent_id}.

Game Rules:
{game_rules}

{group_stats}

Round-by-round results:
{round_results}

The current plan is:
{current_plan}

Choose your move based on this information to reach your goal.
Also decide if you want to continue playing with this opponent or move to the next one.
The winner will get 100000 euros.

{format_instructions}
""")

planner_prompt = ChatPromptTemplate.from_template("""
You are a strategic planner for Player {player_id} in an iterated game tournament, player {player_id} is playing against player {opponent_id}.
The rules of the game are as follows:
{game_rules}

Players stats:
{players_stats}

{group_stats}

Round-by-round results of previous matches:
{history}

Your strategy last time was:
{old_plan}

Feedback from the critic:
{critique}

{planner_instructions}

Create a short plan that will maximize the player's chances of success.
""")

critic_prompt = ChatPromptTemplate.from_template("""
You are a strategic critic evaluating a plan for Player {player_id} in a tournament, player {player_id} is playing against player {opponent_id}.
The rules of the game are as follows:
{game_rules}

Players stats:
{players_stats}

{group_stats}

Round-by-round results of previous matches:
{history}

The current plan is:
{current_plan}

{critic_instructions}

Provide a short review where you provide concrete improvements and suggestions.
""")

# %% 2-vs-1 Discussion Prompts

team_discussion_prompt = ChatPromptTemplate.from_template("""
You are Agent {agent_id} in Team 1, a team of {team_size} agents playing against a single opponent (Team 2).

This is the PRE-GAME DISCUSSION phase (Turn {current_turn} of {max_turns}).

Your teammate is Agent {teammate_id}.

Game Rules:
{game_rules}

Previous discussion (if any):
{discussion_history}

Your teammate's last message (if any):
{teammate_last_message}

Task:
Share your strategic thinking with your teammate. Discuss:
1. What strategy should Team 1 adopt (cooperative, defective, conditional)?
2. How should you coordinate your individual moves?
3. What do you expect from the opponent (Team 2)?
4. Should both team members choose the same action, or different actions?

Remember: You will each choose your own action independently during the game, but you can coordinate strategy now.

Respond with your message to your teammate (2-3 sentences).
""")

adversary_discussion_prompt = ChatPromptTemplate.from_template("""
You are Agent {agent_id}, the sole adversary (Team 2) in a 2-vs-1 strategic game.

You are playing ALONE against a coordinated team of 2 agents (Team 1).

This is your PRE-GAME STRATEGIZING phase (Turn {current_turn} of {max_turns}).

Game Rules:
{game_rules}

Previous thoughts (if any):
{previous_thoughts}

Outcome of the previous game (if any):
{previous_game_outcome}

Task:
Think through your strategy against the coordinated team. Consider:
1. What action (action_a or action_b) should you choose, and why?
2. How might Team 1 coordinate their moves against you?
3. How should you respond to different Team 1 combinations?
4. What is your goal: maximize your own score?

Respond with your strategic thoughts (2-3 sentences).
""")


def format_game_rules(current_round, total_rounds) -> str:
    if current_tournament_condition == "repeated_only":
        group_rules = "- Your goal is to have the highest personal score."
    elif current_tournament_condition == "competition_only":
        group_rules = (f"""- Your goal is to have the highest group score.
        - Your group score is the sum of all the points gathered by the players in your group.
        - At the end of the game, the group with the highest total score gets a {(GROUP_REWARD_MULTIPLIER - 1) * 100}% bonus""")
    else:
        group_rules = (f"""- Your goal is to have the highest group and personal score.
        - Your group score is the sum of all the points gathered by the players in your group.
        - At the end of the game, the group with the highest total score gets a {(GROUP_REWARD_MULTIPLIER - 1) * 100}% bonus""")

    return f"""        - Choose "action_a" or "action_b" each round
        - Players choose their actions at the same time
        - If both choose action_a: 3 points each 
        - If both choose action_b: 0 points each
        - If one chooses action_a, one chooses action_b: action_b player gets 5 points, action_a player gets -1 points
        - You can play at most {MAX_TOTAL_ROUNDS} total rounds across all matches, so far you have played {total_rounds}
        - Each match has a maximum of {MAX_ROUNDS_PER_MATCH} rounds
        - After each round, you can choose to continue with the current opponent or move to the next one, your opponent can do the same
        - Current round: {current_round}
        {group_rules}
        """


def format_match_results(match: MatchState, player_id: int) -> str:
    """Format round-by-round results for a specific player"""
    results = []
    is_player1 = player_id == match.player1_id

    for i, (p1_move, p2_move) in enumerate(match.round_results):
        p1_score, p2_score = match.round_scores[i]
        my_move = p1_move if is_player1 else p2_move
        opp_move = p2_move if is_player1 else p1_move
        my_score = p1_score if is_player1 else p2_score
        opp_score = p2_score if is_player1 else p1_score

        results.append(
            f"Round {i + 1}: You chose {my_move}, opponent chose {opp_move}. Score: +{my_score} for you, +{opp_score} for opponent")

    return "\n".join(results) if results else "No rounds played yet"


def format_player_history(player_id: int, tournament_state: TournamentState, current_match: MatchState) -> str:
    """Format the player's history for the planner/critic"""
    history = []
    for match in tournament_state.matches.values():
        if player_id == match.player1_id or player_id == match.player2_id:
            match_res = format_match_results(match, player_id)
            if match_res != "No rounds played yet":
                if current_tournament_condition == "repeated_only":
                    history.append(f"\nResults of match between player {match.player1_id} and player {match.player2_id}:")
                else:
                    history.append(f"\nResults of match between player {match.player1_id} from group {match.player1_stats.group_id} and player {match.player2_id} from group {match.player2_stats.group_id}:")
                history.append(match_res)

    return "\n".join(history) if history else "No rounds played yet"


def format_group_stats(player_id: int, opponent_id: int, tournament_state: TournamentState) -> str:
    if current_tournament_condition == "repeated_only":
        return ""
    # Get player's group info
    player_group_id = tournament_state.players[player_id].group_id
    if opponent_id == "unknown":
        opponent_group_id = "unknown"
    else:
        opponent_group_id = tournament_state.players[opponent_id].group_id

    # Calculate group rank
    group_scores = [(g_id, g.total_score) for g_id, g in tournament_state.groups.items()]
    group_scores.sort(key=lambda x: x[1], reverse=True)
    group_rank = next((i + 1 for i, (g_id, _) in enumerate(group_scores) if g_id == player_group_id), 0)

    # Get group cooperation rate
    group_coop_rate = tournament_state.groups[player_group_id].avg_cooperation_rate * 100

    group_members = tournament_state.groups[player_group_id].members

    return f"""
Group information:
    - You are Player {player_id} in Group {player_group_id}
    - Your opponent is Player {opponent_id} in Group {opponent_group_id}
    - Your group's current rank: {group_rank} of {NUM_GROUPS}
    - Your group's members are: {["player " + str(i) for i in group_members]}
    """


def format_players_stats(player_id: int, opponent_id: int, state: MatchState):
    is_player1 = player_id == state.player1_id
    my_stats = state.player1_stats if is_player1 else state.player2_stats
    opp_stats = state.player2_stats if is_player1 else state.player1_stats

    total_moves = my_stats.action_a_count + my_stats.action_b_count
    opp_total_moves = opp_stats.action_a_count + opp_stats.action_b_count

    action_a_rate = (my_stats.action_a_count / total_moves * 100) if total_moves > 0 else 'unknown'
    opp_action_a_rate = (opp_stats.action_a_count / opp_total_moves * 100) if opp_total_moves > 0 else 'unknown'
    your_score = my_stats.total_score
    opponent_score = opp_stats.total_score

    return f"""        Your match action_a rate: {action_a_rate} %
        Opponent's match action_a rate: {opp_action_a_rate} %
        Your match score: {your_score}
        Opponent score: {opponent_score}
        """


def format_planner_instructions() -> str:
    if current_tournament_condition == "repeated_only":
        goals = ""
    else:
        goals = "\n4. Individual and group goals\n"
    return f"""Your task is to create a short plan for the player following the rules of the game described in the rules section. Consider for example:
        1. Identifying the action patterns and strategies in the opponent's behavior, when does the opponent pick action_a? when does he pick action_b?
        2. Which actions to pick to counter the opponents strategy
        3. If it is profitable to keep playing with the current opponent or it is better move on
        4. Long term and short term rewards, do you build trust with action_a or betray your opponent with action_b?{goals}
        Be specific, do not give a generic plan, go into details on which actions to choose and when."""


def format_critic_instructions() -> str:
    if current_tournament_condition == "repeated_only":
        goals = ""
    else:
        goals = "6. How the plan balances individual and group goals"
    return f"""Your task is to critically evaluate this plan and suggest improvements. Consider for example:
        1. If the plan follows the rules of the game described above
        2. How well the plan describes the opponent's strategy so far
        3. Whether the plan adequately counters the opponent's strategy.
        4. Whether the plan balances short-term and long-term objectives
        5. Is the plan specific? does it give a concrete plan that goes into details on which actions to pick?
        {goals}"""


# %% 2-vs-1 Game Specific Functions

def format_game_rules_2v1(current_round, total_rounds) -> str:
    """Format game rules for 2-vs-1 team game."""
    return f"""
    GAME STRUCTURE:
    - You are in a 2-vs-1 game
    - Team 1 has 2 agents (you and your teammate)
    - Team 2 has 1 agent (your opponent)

    ACTIONS:
    - Each Team 1 agent independently chooses "action_a" or "action_b" each round
    - Team 2 agent chooses one action that applies to both Team 1 members
    - Possible Team 1 combinations: (a,a), (a,b), (b,a), (b,b)
    - Possible Team 2 actions: a, b

    PAYOFFS:
    - Team 1 receives a TEAM SCORE (not individual scores)
    - Your team's success depends on coordination and strategy
    - Payoff depends on: (your_move, teammate_move, opponent_move)
    - See full payoff matrix for details

    GAME FLOW:
    - Current round: {current_round}
    - Total rounds played: {total_rounds}
    - Max rounds per match: {MAX_ROUNDS_PER_MATCH}
    - Max total rounds: {MAX_TOTAL_ROUNDS}

    GOAL:
    - Maximize Team 1's total score
    - Coordinate with your teammate for optimal outcomes
    """


def run_team_discussion(state: MatchState, tournament_state: TournamentState, max_turns: int) -> str:
    """
    Run multi-turn discussion between Team 1 members before game starts.

    Returns: Shared team plan summary
    """
    llm = LLM
    team1_ids = state.team1_member_ids

    print(f"\n{'='*60}")
    print(f"TEAM 1 DISCUSSION PHASE ({max_turns} turns)")
    print(f"{'='*60}")

    discussion_history = []

    for turn in range(max_turns):
        for agent_id in team1_ids:
            teammate_id = [tid for tid in team1_ids if tid != agent_id][0]

            # Get teammate's last message
            teammate_last = ""
            if discussion_history:
                for msg in reversed(discussion_history):
                    if msg.get('agent_id') == teammate_id:
                        teammate_last = msg.get('message', '')
                        break

            # Format discussion history
            history_text = "\n".join([
                f"Agent {msg['agent_id']}: {msg['message']}"
                for msg in discussion_history
            ]) if discussion_history else "No previous discussion."

            # Generate discussion message
            prompt = team_discussion_prompt.format(
                agent_id=agent_id,
                team_size=TEAM1_SIZE,
                teammate_id=teammate_id,
                current_turn=turn + 1,
                max_turns=max_turns,
                game_rules=format_game_rules_2v1(0, 0),
                discussion_history=history_text,
                teammate_last_message=teammate_last if teammate_last else "No message yet."
            )

            response = llm.invoke(prompt)
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

            discussion_history.append({
                'agent_id': agent_id,
                'turn': turn,
                'message': response
            })

            print(f"\nTurn {turn+1} - Agent {agent_id}:")
            print(f"  {response}")

    # Store discussion history in match state
    state.discussion_history = discussion_history

    # Generate shared team plan summary
    summary_prompt = f"""
    Based on this team discussion:
    {chr(10).join([f"Agent {msg['agent_id']}: {msg['message']}" for msg in discussion_history])}

    Summarize the team's agreed strategy in 2-3 sentences.
    """

    shared_plan = llm.invoke(summary_prompt)
    shared_plan = re.sub(r"<think>.*?</think>", "", shared_plan, flags=re.DOTALL)
    state.team1_shared_plan = shared_plan

    print(f"\n{'='*60}")
    print(f"SHARED TEAM PLAN: {shared_plan}")
    print(f"{'='*60}\n")

    return shared_plan


def format_team_match_results(match: MatchState, agent_id: int) -> str:
    """Format match results from Team 1 member's perspective."""
    if not match.round_results:
        return "No rounds played yet."

    results = []
    teammate_id = [tid for tid in match.team1_member_ids if tid != agent_id][0]

    for i, round_data in enumerate(match.round_results):
        if isinstance(round_data, dict):  # Team game format
            my_move = round_data['team1'][agent_id]
            teammate_move = round_data['team1'][teammate_id]
            opponent_move = round_data['team2']

            team_score, opponent_score = match.round_scores[i]

            results.append(
                f"Round {i+1}: You={my_move}, Teammate={teammate_move}, Opponent={opponent_move} | "
                f"Team Score: +{team_score}, Opponent: +{opponent_score}"
            )

    return "\n".join(results) if results else "No rounds played yet."


def format_team_match_results_team2(match: MatchState) -> str:
    """Format match results from Team 2's perspective."""
    if not match.round_results:
        return "No rounds played yet."

    results = []
    agent1_id = match.team1_member_ids[0]
    agent2_id = match.team1_member_ids[1]

    for i, round_data in enumerate(match.round_results):
        if isinstance(round_data, dict):  # Team game format
            agent1_move = round_data['team1'][agent1_id]
            agent2_move = round_data['team1'][agent2_id]
            my_move = round_data['team2']

            team1_score, my_score = match.round_scores[i]

            results.append(
                f"Round {i+1}: Opponent1={agent1_move}, Opponent2={agent2_move}, You={my_move} | "
                f"Your Score: +{my_score}, Opponents' Team: +{team1_score}"
            )

    return "\n".join(results) if results else "No rounds played yet."


def run_adversary_discussion(state: MatchState, previous_game_outcome: str, max_turns: int) -> str:
    """
    Run a solo strategizing phase for the Team 2 adversary before each game.
    The adversary thinks through their strategy across max_turns iterations.

    Returns: Adversary's strategy plan summary
    """
    llm = LLM
    agent_id = state.team2_member_id

    print(f"\n{'='*60}")
    print(f"TEAM 2 (ADVERSARY) STRATEGIZING PHASE ({max_turns} turns)")
    print(f"{'='*60}")

    thoughts = []

    for turn in range(max_turns):
        previous_text = "\n".join([
            f"Turn {t['turn']+1}: {t['message']}" for t in thoughts
        ]) if thoughts else "No previous thoughts yet."

        prompt = adversary_discussion_prompt.format(
            agent_id=agent_id,
            current_turn=turn + 1,
            max_turns=max_turns,
            game_rules=format_game_rules_2v1(0, 0),
            previous_thoughts=previous_text,
            previous_game_outcome=previous_game_outcome,
        )

        response = llm.invoke(prompt)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

        thoughts.append({'turn': turn, 'message': response})

        print(f"\nTurn {turn+1} - Adversary Agent {agent_id}:")
        print(f"  {response}")

    # The final thought becomes the adversary's plan
    final_plan = thoughts[-1]['message'] if thoughts else "No strategy formed."
    state.team2_stats.plan = final_plan

    print(f"\n{'='*60}")
    print(f"ADVERSARY PLAN: {final_plan}")
    print(f"{'='*60}\n")

    return final_plan


def compute_empirical_frequencies(match: MatchState) -> Dict[str, Any]:
    """
    Compute empirical action frequencies and joint outcome probabilities for a match.

    Returns a dict with:
    - 'team1_agent1_action_a_rate': fraction of action_a for Team 1 agent 1
    - 'team1_agent2_action_a_rate': fraction of action_a for Team 1 agent 2
    - 'team2_action_a_rate': fraction of action_a for Team 2
    - 'joint_outcome_counts': raw counts for all 8 (a1, a2, t2) combinations
    - 'joint_outcome_frequencies': normalized frequencies for all 8 combinations
    - 'team1_joint_counts': counts for Team 1 joint actions (a,a), (a,b), (b,a), (b,b)
    - 'team1_joint_frequencies': normalized frequencies for Team 1 joint actions
    """
    if not match.round_results:
        return {}

    agent1_id = match.team1_member_ids[0]
    agent2_id = match.team1_member_ids[1]

    n = len(match.round_results)

    # Marginal action counts
    a1_a_count = sum(1 for r in match.round_results if isinstance(r, dict) and r['team1'][agent1_id] == 'action_a')
    a2_a_count = sum(1 for r in match.round_results if isinstance(r, dict) and r['team1'][agent2_id] == 'action_a')
    t2_a_count = sum(1 for r in match.round_results if isinstance(r, dict) and r['team2'] == 'action_a')

    # Joint outcome counts over all 8 combinations
    combinations_keys = [
        ('action_a', 'action_a', 'action_a'),
        ('action_a', 'action_a', 'action_b'),
        ('action_a', 'action_b', 'action_a'),
        ('action_a', 'action_b', 'action_b'),
        ('action_b', 'action_a', 'action_a'),
        ('action_b', 'action_a', 'action_b'),
        ('action_b', 'action_b', 'action_a'),
        ('action_b', 'action_b', 'action_b'),
    ]

    joint_counts = {}
    for key in combinations_keys:
        a1m, a2m, t2m = key
        count = sum(
            1 for r in match.round_results
            if isinstance(r, dict)
            and r['team1'][agent1_id] == a1m
            and r['team1'][agent2_id] == a2m
            and r['team2'] == t2m
        )
        label = f"({a1m.replace('action_', '')},{a2m.replace('action_', '')},{t2m.replace('action_', '')})"
        joint_counts[label] = count

    joint_frequencies = {k: v / n for k, v in joint_counts.items()}

    # Team 1 joint action counts (collapsing Team 2)
    team1_joint_labels = {'(a,a)': ('action_a', 'action_a'), '(a,b)': ('action_a', 'action_b'),
                          '(b,a)': ('action_b', 'action_a'), '(b,b)': ('action_b', 'action_b')}
    team1_joint_counts = {}
    for label, (a1m, a2m) in team1_joint_labels.items():
        team1_joint_counts[label] = sum(
            1 for r in match.round_results
            if isinstance(r, dict)
            and r['team1'][agent1_id] == a1m
            and r['team1'][agent2_id] == a2m
        )
    team1_joint_frequencies = {k: v / n for k, v in team1_joint_counts.items()}

    return {
        'num_rounds': n,
        'team1_agent1_action_a_rate': a1_a_count / n,
        'team1_agent2_action_a_rate': a2_a_count / n,
        'team2_action_a_rate': t2_a_count / n,
        'joint_outcome_counts': joint_counts,
        'joint_outcome_frequencies': joint_frequencies,
        'team1_joint_counts': team1_joint_counts,
        'team1_joint_frequencies': team1_joint_frequencies,
    }


# %% LLM calls


def generate_next_move(player_id: int, state: MatchState, opponent_id: int, tournament_state: TournamentState) -> Dict[str, Any]:
    parser = PydanticOutputParser(pydantic_object=Action)
    llm = LLM

    is_player1 = player_id == state.player1_id

    plan = state.player1_stats.plan if is_player1 else state.player2_stats.plan

    if is_player1:
        total_rounds = tournament_state.players[state.player1_id].action_a_count + tournament_state.players[state.player1_id].action_b_count + state.player1_stats.action_a_count + state.player1_stats.action_b_count
    else:
        total_rounds = tournament_state.players[state.player2_id].action_a_count + tournament_state.players[state.player2_id].action_b_count + state.player2_stats.action_a_count + state.player2_stats.action_b_count


    if state.is_first_interaction:
        opponent_id = "unknown"

    prompt = player_prompt.format(
        player_id=player_id,
        opponent_id=opponent_id,
        game_rules=format_game_rules(state.current_round, total_rounds),
        round_results=format_match_results(state, player_id) if PLANNING_FREQUENCY != 0 else format_player_history(
            player_id, tournament_state, state),
        group_stats=format_group_stats(player_id, opponent_id, tournament_state),
        current_plan=plan,
        format_instructions=parser.get_format_instructions(),
    )

    if COOPERATE_DEFECT_PROMPT:
        prompt = prompt.replace('action_a', 'cooperate').replace('action_b', 'defect')

    response = llm.invoke(prompt)

    try:
        response = response.lower()
        response = response.replace('cooperate', 'action_a').replace('defect', 'action_b')
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        parsed_output = parser.parse(response)
        move = parsed_output.move
        reasoning = parsed_output.reasoning
        keep_playing = parsed_output.keep_playing
    except Exception as e:
        if re.search(r'(answer|final).*(defect|action_b)', response):
            move = "action_b"
            reasoning = response
            keep_playing = True  # Default to continue playing
        elif re.search(r'(answer|final).*(cooperate|action_a)', response):
            move = "action_a"
            reasoning = response
            keep_playing = True  # Default to continue playing
        else:
            print(f"Error parsing Player {player_id} output: {e}")
            move = random.choice(["action_a", "action_b"])
            reasoning = "Error in parsing, random move chosen."
            keep_playing = True  # Default to continue playing

    print(f"Player {player_id} chose {move} and {'wants to continue' if keep_playing else 'wants to move on'}. Reasoning: {reasoning}")
    return {"move": move, "reasoning": reasoning, "keep_playing": keep_playing}


def generate_team1_move(agent_id: int, state: MatchState, tournament_state: TournamentState) -> Dict[str, Any]:
    """Generate move for a Team 1 member with team context."""
    parser = PydanticOutputParser(pydantic_object=Action)
    llm = LLM

    # Get teammate info
    teammate_id = [tid for tid in state.team1_member_ids if tid != agent_id][0]

    # Get agent's current plan
    agent_stats = state.team1_stats[agent_id]
    plan = agent_stats.plan

    # Calculate total rounds
    total_rounds = (tournament_state.players[agent_id].action_a_count +
                   tournament_state.players[agent_id].action_b_count +
                   agent_stats.action_a_count + agent_stats.action_b_count)

    # Format team context
    team_context = f"""
    TEAM INFORMATION:
    - Your teammate: Agent {teammate_id}
    - Shared team plan: {state.team1_shared_plan}
    - Teammate's plan: {state.team1_stats[teammate_id].plan}

    Previous round results (if any):
    {format_team_match_results(state, agent_id)}
    """

    prompt = f"""
You are Agent {agent_id} in Team 1, playing against Team 2 (single opponent).

{format_game_rules_2v1(state.current_round, total_rounds)}

{team_context}

Your current plan:
{plan}

Choose your move (action_a or action_b) based on:
1. Your shared team strategy
2. Your teammate's likely move
3. The opponent's past behavior
4. Maximizing team score

Also decide if you want to continue playing or move to the next opponent.

{parser.get_format_instructions()}
"""

    response = llm.invoke(prompt)

    # Parse response (same error handling as original)
    try:
        response = response.lower()
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        parsed_output = parser.parse(response)
        move = parsed_output.move
        reasoning = parsed_output.reasoning
        keep_playing = parsed_output.keep_playing
    except Exception as e:
        if re.search(r'(answer|final).*(action_b)', response):
            move = "action_b"
            reasoning = response
            keep_playing = True
        elif re.search(r'(answer|final).*(action_a)', response):
            move = "action_a"
            reasoning = response
            keep_playing = True
        else:
            print(f"Error parsing Agent {agent_id} output: {e}")
            move = random.choice(["action_a", "action_b"])
            reasoning = "Error in parsing, random move chosen."
            keep_playing = True

    print(f"Team 1 Agent {agent_id} chose {move} and {'wants to continue' if keep_playing else 'wants to move on'}. Reasoning: {reasoning}")
    return {"move": move, "reasoning": reasoning, "keep_playing": keep_playing}


def generate_team2_move(state: MatchState, tournament_state: TournamentState) -> Dict[str, Any]:
    """Generate move for Team 2 (single agent facing a team)."""
    parser = PydanticOutputParser(pydantic_object=Action)
    llm = LLM

    agent_id = state.team2_member_id
    plan = state.team2_stats.plan

    # Calculate total rounds
    total_rounds = (tournament_state.players[agent_id].action_a_count +
                   tournament_state.players[agent_id].action_b_count +
                   state.team2_stats.action_a_count + state.team2_stats.action_b_count)

    prompt = f"""
You are Agent {agent_id} in Team 2, playing ALONE against a team of 2 opponents (Team 1).

{format_game_rules_2v1(state.current_round, total_rounds)}

IMPORTANT: You face BOTH Team 1 agents with a single action choice.
- You can see their individual past moves
- Your action applies to both of them
- They may coordinate their strategies

Previous rounds:
{format_team_match_results_team2(state)}

Your current plan:
{plan}

Choose your move (action_a or action_b) to maximize your score.

{parser.get_format_instructions()}
"""

    response = llm.invoke(prompt)

    # Parse response (same error handling)
    try:
        response = response.lower()
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        parsed_output = parser.parse(response)
        move = parsed_output.move
        reasoning = parsed_output.reasoning
        keep_playing = parsed_output.keep_playing
    except Exception as e:
        if re.search(r'(answer|final).*(action_b)', response):
            move = "action_b"
            reasoning = response
            keep_playing = True
        elif re.search(r'(answer|final).*(action_a)', response):
            move = "action_a"
            reasoning = response
            keep_playing = True
        else:
            print(f"Error parsing Team 2 Agent {agent_id} output: {e}")
            move = random.choice(["action_a", "action_b"])
            reasoning = "Error in parsing, random move chosen."
            keep_playing = True

    print(f"Team 2 Agent {agent_id} chose {move} and {'wants to continue' if keep_playing else 'wants to move on'}. Reasoning: {reasoning}")
    return {"move": move, "reasoning": reasoning, "keep_playing": keep_playing}


def generate_plan(player_id: int, state: MatchState, opponent_id: int, tournament_state: TournamentState, old_plan: str,
                  critique: str) -> str:
    """Generate a strategic plan for the player"""
    llm = LLM

    is_player1 = player_id == state.player1_id

    if is_player1:
        total_rounds = tournament_state.players[state.player1_id].action_a_count + tournament_state.players[state.player1_id].action_b_count + state.player1_stats.action_a_count + state.player1_stats.action_b_count
    else:
        total_rounds = tournament_state.players[state.player2_id].action_a_count + tournament_state.players[state.player2_id].action_b_count + state.player2_stats.action_a_count + state.player2_stats.action_b_count


    prompt = planner_prompt.format(
        player_id=player_id,
        opponent_id=opponent_id,
        game_rules=format_game_rules(state.current_round, total_rounds),
        players_stats=format_players_stats(player_id, opponent_id, state),
        history=format_player_history(player_id, tournament_state, state),
        group_stats=format_group_stats(player_id, opponent_id, tournament_state),
        planner_instructions=format_planner_instructions(),
        old_plan=old_plan,
        critique=critique,
    )

    if COOPERATE_DEFECT_PROMPT:
        prompt = prompt.replace('action_a', 'cooperate').replace('action_b', 'defect')

    response = llm.invoke(prompt)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return response


def critique_plan(player_id: int, state: MatchState, opponent_id: int, tournament_state: TournamentState,
                  plan: str) -> str:
    """Critique the player's strategic plan"""
    llm = LLM

    is_player1 = player_id == state.player1_id

    if is_player1:
        total_rounds = tournament_state.players[state.player1_id].action_a_count + tournament_state.players[
            state.player1_id].action_b_count + state.player1_stats.action_a_count + state.player1_stats.action_b_count
    else:
        total_rounds = tournament_state.players[state.player2_id].action_a_count + tournament_state.players[
            state.player2_id].action_b_count + state.player2_stats.action_a_count + state.player2_stats.action_b_count

    prompt = critic_prompt.format(
        player_id=player_id,
        opponent_id=opponent_id,
        game_rules=format_game_rules(state.current_round, total_rounds),
        players_stats=format_players_stats(player_id, opponent_id, state),
        history=format_player_history(player_id, tournament_state, state),
        group_stats=format_group_stats(player_id, opponent_id, tournament_state),
        critic_instructions=format_critic_instructions(),
        current_plan=plan,
    )

    if COOPERATE_DEFECT_PROMPT:
        prompt = prompt.replace('action_a', 'cooperate').replace('action_b', 'defect')

    response = llm.invoke(prompt)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return response


def plan_and_critique(state: MatchState, tournament_state: TournamentState, player_id: int, opponent_id: int,
                      max_iterations: int = 1) -> (str, str):
    """Generate and refine a strategy through planning and critique"""
    is_player1 = player_id == state.player1_id
    critique = state.player1_stats.critique if is_player1 else state.player2_stats.critique
    plan = state.player1_stats.plan if is_player1 else state.player2_stats.plan

    for i in range(max_iterations):
        plan = generate_plan(player_id=player_id, state=state, opponent_id=opponent_id,
                             tournament_state=tournament_state, old_plan=plan, critique=critique)
        critique = critique_plan(player_id=player_id, state=state, opponent_id=opponent_id,
                                 tournament_state=tournament_state, plan=plan)

        print(f"Player {player_id}'s plan iteration {i + 1} generating improved plan.")

    final_plan = generate_plan(player_id=player_id, state=state, opponent_id=opponent_id,
                               tournament_state=tournament_state, old_plan=plan, critique=critique)

    # Print the final plan
    if player_id == state.player1_id:
        print(f"Player {player_id}'s strategy: {final_plan}")
    if player_id == state.player2_id:
        print(f"Player {player_id}'s strategy: {final_plan}")

    return plan, critique


def run_meta_prompt(player_id: int, state: MatchState, opponent_id: int, tournament_state: TournamentState) -> Dict[
    str, Any]:
    parser = PydanticOutputParser(pydantic_object=MetaPromptFields)
    llm = LLM

    is_player1 = player_id == state.player1_id
    random_action_1 = random.choice(["action_a", "action_b"])
    random_action_2 = random.choice(["action_a", "action_b"])
    random_round_1 = random.choice(range(1, state.current_round))
    random_round_2 = random.choice(range(1, state.current_round))

    if is_player1:
        total_rounds = tournament_state.players[state.player1_id].action_a_count + tournament_state.players[
            state.player1_id].action_b_count + state.player1_stats.action_a_count + state.player1_stats.action_b_count
    else:
        total_rounds = tournament_state.players[state.player2_id].action_a_count + tournament_state.players[
            state.player2_id].action_b_count + state.player2_stats.action_a_count + state.player2_stats.action_b_count

    response = llm.invoke(
        meta_prompt.format(
            player_id=player_id,
            opponent_id=opponent_id,
            game_rules=format_game_rules(state.current_round, total_rounds),
            round_results=format_match_results(state, player_id),
            group_stats=format_group_stats(player_id, opponent_id, tournament_state),
            random_action_1=random_action_1,
            random_action_2=random_action_2,
            random_round_1=random_round_1,
            random_round_2=random_round_2,
            format_instructions=parser.get_format_instructions(),
        )
    )

    try:
        response = response.replace('cooperate', 'action_a').replace('defect', 'action_b')
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        parsed_output = parser.parse(response)
        min_max = parsed_output.min_max
        actions = parsed_output.actions
        payoff = parsed_output.payoff
        round = parsed_output.round
        action = parsed_output.action
        points = parsed_output.points
        num_actions = parsed_output.num_actions
        num_points = parsed_output.num_points
        tft = parsed_output.tft
        forgiving = parsed_output.forgiving
    except Exception as e:
        print(f"Error parsing Player {player_id} output: {e}")
        min_max = (0, 0)
        actions = ['']
        payoff = 0
        round = 0
        action = 'action_a'
        points = 0
        num_actions = 0
        num_points = 0
        tft = False
        forgiving = False

    # Evaluate answers to the meta prompt
    min_max_score = int(min_max == (-1, 5))
    if actions == ['action_a', 'action_b'] or actions == ['action_b', 'action_a'] or actions == ['action a',
                                                                                                 'action b'] or actions == [
        'action b', 'action a']:
        actions_score = 1
    else:
        actions_score = 0
    payoff_score = PAYOFFS[(random_action_1, random_action_2)][0] == payoff
    round_score = state.current_round == round
    action_score = state.round_results[random_round_1 - 1][1] == action if is_player1 else \
    state.round_results[random_round_1 - 1][0] == action
    points_score = state.round_scores[random_round_2 - 1][1] == points if is_player1 else \
    state.round_scores[random_round_2 - 1][0] == points
    if random_action_1 == 'action_a':
        num_actions_score = state.player2_stats.action_a_count == num_actions if is_player1 else state.player1_stats.action_a_count == num_actions
    else:
        num_actions_score = state.player1_stats.action_b_count == num_actions if is_player1 else state.player2_stats.action_b_count == num_actions
    num_points_score = state.player2_stats.total_score == num_points if is_player1 else state.player1_stats.total_score == num_points
    true_tft = tournament_state.players[opponent_id].SFEM[-1]['tft'] > 0.9
    tft_score = true_tft == tft
    true_forgiving = tournament_state.players[opponent_id].traits[-1]['forgiving'] > 0.7
    forgiving_score = true_forgiving == forgiving

    return {
        'min_max': min_max_score,
        'actions': actions_score,
        'payoff': payoff_score,
        'round': round_score,
        'action': action_score,
        'points': points_score,
        'num_actions': num_actions_score,
        'num_points': num_points_score,
        'tft': tft_score,
        'forgiving': forgiving_score,
    }


# %% Game logic


def initialize_groups_and_players():
    groups = {}
    players = {}
    player_id = 0

    for g in range(NUM_GROUPS):
        groups[g] = GroupStats(group_id=g, members=[])

        for _ in range(GROUP_SIZE):
            players[player_id] = PlayerStats(group_id=g)
            groups[g].members.append(player_id)
            player_id += 1

    return players, groups


def print_match_results(match: MatchState):
    print(f"\nMatch between Player {match.player1_id} and Player {match.player2_id}:")
    print(f"Player {match.player1_id} Score: {match.player1_stats.total_score}")
    print(f"Player {match.player2_id} Score: {match.player2_stats.total_score}")
    p1_action_a_rate = (match.player1_stats.action_a_count / (
                match.player1_stats.action_a_count + match.player1_stats.action_b_count) * 100)
    p2_action_a_rate = (match.player2_stats.action_a_count / (
                match.player2_stats.action_a_count + match.player2_stats.action_b_count) * 100)
    print(f"Player {match.player1_id} Action A Rate: {p1_action_a_rate:.1f}%")
    print(f"Player {match.player2_id} Action A Rate: {p2_action_a_rate:.1f}%")
    print("\n" + "-" * 50)


def player1_move(state: TournamentState):
    match = state.matches[state.current_match_idx]
    result = generate_next_move(match.player1_id, match, match.player2_id, state)
    match.player1_move = result['move']
    match.player1_keep_playing = result['keep_playing']
    return {"matches": {state.current_match_idx: match}}


def player2_move(state: TournamentState):
    match = state.matches[state.current_match_idx]
    result = generate_next_move(match.player2_id, match, match.player1_id, state)
    match.player2_move = result['move']
    match.player2_keep_playing = result['keep_playing']
    return {"matches": {state.current_match_idx: match}}


def player1_plan(state: TournamentState):
    match = state.matches[state.current_match_idx]
    total_rounds = state.players[match.player1_id].action_a_count + state.players[match.player1_id].action_b_count + match.player1_stats.action_a_count + match.player1_stats.action_b_count
    if PLANNING_FREQUENCY != 0 and total_rounds % PLANNING_FREQUENCY == 0 and total_rounds != 0:
        match.player1_stats.plan, match.player1_stats.critique = plan_and_critique(match, state, match.player1_id,
                                                                                   match.player2_id,
                                                                                   CRITIQUE_ITERATIONS)
        return {"matches": {state.current_match_idx: match}}


def player2_plan(state: TournamentState):
    match = state.matches[state.current_match_idx]
    total_rounds = state.players[match.player2_id].action_a_count + state.players[match.player2_id].action_b_count + match.player2_stats.action_a_count + match.player2_stats.action_b_count
    if PLANNING_FREQUENCY != 0 and total_rounds % PLANNING_FREQUENCY == 0 and total_rounds != 0:
        match.player2_stats.plan, match.player2_stats.critique = plan_and_critique(match, state, match.player2_id,
                                                                                   match.player1_id,
                                                                                   CRITIQUE_ITERATIONS)
        return {"matches": {state.current_match_idx: match}}


def start_round(state: TournamentState):
    pass


def play_round(state: TournamentState):
    match = state.matches[state.current_match_idx]

    # Get moves
    p1_move = match.player1_move
    p2_move = match.player2_move

    # Calculate payoffs
    p1_payoff, p2_payoff = PAYOFFS[(p1_move, p2_move)]

    # Update round results
    match.round_results.append((p1_move, p2_move))
    match.round_scores.append((p1_payoff, p2_payoff))

    # Track first interaction cooperation
    if match.is_first_interaction:
        state.record_first_interaction(match.player1_id, match.player2_id, p1_move, p2_move)
        match.is_first_interaction = False

    # Update stats
    match.player1_stats.total_score += p1_payoff
    match.player2_stats.total_score += p2_payoff

    if p1_move == "action_a":
        match.player1_stats.action_a_count += 1
    else:
        match.player1_stats.action_b_count += 1

    if p2_move == "action_a":
        match.player2_stats.action_a_count += 1
    else:
        match.player2_stats.action_b_count += 1

    # keep track of some statistics
    p1_total_moves = match.player1_stats.action_a_count + match.player1_stats.action_b_count
    p2_total_moves = match.player2_stats.action_a_count + match.player2_stats.action_b_count

    # Calculate and save action rates
    p1_action_a_rate = (match.player1_stats.action_a_count / p1_total_moves) * 100
    p2_action_a_rate = (match.player2_stats.action_a_count / p2_total_moves) * 100
    match.player1_stats.action_a_rate_history.append(p1_action_a_rate)
    match.player2_stats.action_a_rate_history.append(p2_action_a_rate)

    # Track scores
    match.player1_stats.score_history.append(match.player1_stats.total_score)
    match.player2_stats.score_history.append(match.player2_stats.total_score)

    # Update move history
    match.player1_stats.move_history.append(p1_move)
    match.player2_stats.move_history.append(p2_move)

    match.current_round += 1
    state.round_number += 1

    # Check if either player wants to end the match or if limits are reached
    p1_total_rounds = state.players[match.player1_id].action_a_count + state.players[match.player1_id].action_b_count + match.player1_stats.action_a_count + match.player1_stats.action_b_count
    p2_total_rounds = state.players[match.player2_id].action_a_count + state.players[match.player2_id].action_b_count + match.player2_stats.action_a_count + match.player2_stats.action_b_count

    end_match = (
            (not match.player1_keep_playing or not match.player2_keep_playing) or  # Either player wants to end
            match.current_round > MAX_ROUNDS_PER_MATCH or  # Max rounds per match reached
            p1_total_rounds >= MAX_TOTAL_ROUNDS or  # Player 1 reached max total rounds
            p2_total_rounds >= MAX_TOTAL_ROUNDS  # Player 2 reached max total rounds
    )

    old_match_idx = state.current_match_idx
    if end_match:
        match.completed = True
        # Add all the player stats from that match to the player stats across all matches
        state.update_player_stats(match.player1_id, match.player1_stats, match.player2_stats)
        state.update_player_stats(match.player2_id, match.player2_stats, match.player2_stats)

        # Update group stats
        state.update_group_stats()

        print_match_results(match)

        # Show why the match ended
        if not match.player1_keep_playing:
            print(f"Match ended because Player {match.player1_id} chose to move on")
        if not match.player2_keep_playing:
            print(f"Match ended because Player {match.player2_id} chose to move on")
        if match.current_round > MAX_ROUNDS_PER_MATCH:
            print(f"Match ended because maximum rounds per match ({MAX_ROUNDS_PER_MATCH}) was reached")
        if p1_total_rounds >= MAX_TOTAL_ROUNDS:
            print(f"Match ended because Player {match.player1_id} reached maximum total rounds ({MAX_TOTAL_ROUNDS})")
        if p2_total_rounds >= MAX_TOTAL_ROUNDS:
            print(f"Match ended because Player {match.player2_id} reached maximum total rounds ({MAX_TOTAL_ROUNDS})")

        # Updated traits and affinities
        state.update_nth_stats(match.player1_id, match.player1_stats, match.player2_stats)
        state.update_nth_stats(match.player2_id, match.player2_stats, match.player1_stats)

        # Update meta prompt results (only for player 0)
        if match.player1_id == 0:
            p1_meta_prompt_results = run_meta_prompt(match.player1_id, match, match.player2_id, state)
            state.players[match.player1_id].meta_prompt_results.append(p1_meta_prompt_results)

        state.current_match_idx += 1
        if state.current_match_idx < len(state.matches.keys()):
            # Copy the plan and critique to keep for the next match
            new_match = state.matches[state.current_match_idx]
            new_match.player1_stats.plan = state.players[new_match.player1_id].plan
            new_match.player1_stats.critique = state.players[new_match.player1_id].critique
            new_match.player2_stats.plan = state.players[new_match.player2_id].plan
            new_match.player2_stats.critique = state.players[new_match.player2_id].critique
            state.matches[state.current_match_idx] = new_match
        # Print match results
        print(f"End of match {state.current_match_idx} out of {len(state.matches)}")

    return {
        "matches": {old_match_idx: match},
        "current_match_idx": state.current_match_idx,
        "round_number": state.round_number,
        "first_interaction_coop_intragroup": state.first_interaction_coop_intragroup,
        "first_interaction_coop_intergroup": state.first_interaction_coop_intergroup,
    }


def should_continue(state: TournamentState) -> str:
    if state.current_match_idx >= len(state.matches.keys()):
        return END

    current_match = state.matches[state.current_match_idx]

    if not current_match.completed:
        return "start_round"

    return "start_round"


# %% 2-vs-1 Workflow Nodes

def start_round_2v1(state: TournamentState):
    """Start a round in 2-vs-1 game."""
    pass  # Just a synchronization point


def team_discussion_node(state: TournamentState):
    """Run team discussion (only on first round of each game)."""
    match = state.matches[state.current_match_idx]
    if match.current_round == 1 and DISCUSSION_TURNS > 0:
        run_team_discussion(match, state, DISCUSSION_TURNS)
    return {"matches": {state.current_match_idx: match}}


def adversary_discussion_node(state: TournamentState):
    """Run adversary solo strategizing phase (only on first round of each game)."""
    match = state.matches[state.current_match_idx]
    if match.current_round == 1 and ADVERSARY_DISCUSSION_TURNS > 0:
        # Summarise previous game outcome for context (empty string on the first ever game)
        previous_game_outcome = match.team2_stats.plan if match.team2_stats.plan != "No plan yet" else "No previous game."
        run_adversary_discussion(match, previous_game_outcome, ADVERSARY_DISCUSSION_TURNS)
    return {"matches": {state.current_match_idx: match}}


def team1_agent1_plan(state: TournamentState):
    """Planning for Team 1 Agent 1."""
    match = state.matches[state.current_match_idx]
    agent_id = match.team1_member_ids[0]
    total_rounds = (state.players[agent_id].action_a_count +
                   state.players[agent_id].action_b_count +
                   match.team1_stats[agent_id].action_a_count +
                   match.team1_stats[agent_id].action_b_count)

    if PLANNING_FREQUENCY != 0 and total_rounds % PLANNING_FREQUENCY == 0 and total_rounds != 0:
        opponent_id = match.team2_member_id
        match.team1_stats[agent_id].plan, match.team1_stats[agent_id].critique = \
            plan_and_critique(match, state, agent_id, opponent_id, CRITIQUE_ITERATIONS)

    return {"matches": {state.current_match_idx: match}}


def team1_agent1_move(state: TournamentState):
    """Move generation for Team 1 Agent 1."""
    match = state.matches[state.current_match_idx]
    agent_id = match.team1_member_ids[0]
    result = generate_team1_move(agent_id, match, state)
    match.team1_moves[agent_id] = result['move']
    match.team1_keep_playing[agent_id] = result['keep_playing']
    return {"matches": {state.current_match_idx: match}}


def team1_agent2_plan(state: TournamentState):
    """Planning for Team 1 Agent 2."""
    match = state.matches[state.current_match_idx]
    agent_id = match.team1_member_ids[1]
    total_rounds = (state.players[agent_id].action_a_count +
                   state.players[agent_id].action_b_count +
                   match.team1_stats[agent_id].action_a_count +
                   match.team1_stats[agent_id].action_b_count)

    if PLANNING_FREQUENCY != 0 and total_rounds % PLANNING_FREQUENCY == 0 and total_rounds != 0:
        opponent_id = match.team2_member_id
        match.team1_stats[agent_id].plan, match.team1_stats[agent_id].critique = \
            plan_and_critique(match, state, agent_id, opponent_id, CRITIQUE_ITERATIONS)

    return {"matches": {state.current_match_idx: match}}


def team1_agent2_move(state: TournamentState):
    """Move generation for Team 1 Agent 2."""
    match = state.matches[state.current_match_idx]
    agent_id = match.team1_member_ids[1]
    result = generate_team1_move(agent_id, match, state)
    match.team1_moves[agent_id] = result['move']
    match.team1_keep_playing[agent_id] = result['keep_playing']
    return {"matches": {state.current_match_idx: match}}


def team2_plan(state: TournamentState):
    """Planning for Team 2."""
    match = state.matches[state.current_match_idx]
    agent_id = match.team2_member_id
    total_rounds = (state.players[agent_id].action_a_count +
                   state.players[agent_id].action_b_count +
                   match.team2_stats.action_a_count +
                   match.team2_stats.action_b_count)

    if PLANNING_FREQUENCY != 0 and total_rounds % PLANNING_FREQUENCY == 0 and total_rounds != 0:
        # Team 2 plans against both Team 1 members
        opponent_id = match.team1_member_ids[0]  # Reference one for now
        match.team2_stats.plan, match.team2_stats.critique = \
            plan_and_critique(match, state, agent_id, opponent_id, CRITIQUE_ITERATIONS)

    return {"matches": {state.current_match_idx: match}}


def team2_move_node(state: TournamentState):
    """Move generation for Team 2."""
    match = state.matches[state.current_match_idx]
    result = generate_team2_move(match, state)
    match.team2_move = result['move']
    match.team2_keep_playing = result['keep_playing']
    return {"matches": {state.current_match_idx: match}}


def print_match_results_2v1(match: MatchState):
    """Print results for 2-vs-1 match."""
    print(f"\n{'='*70}")
    print(f"MATCH RESULTS: Team 1 vs Team 2")
    print(f"{'='*70}")

    agent1_id = match.team1_member_ids[0]
    agent2_id = match.team1_member_ids[1]
    team2_id = match.team2_member_id

    agent1_stats = match.team1_stats[agent1_id]
    agent2_stats = match.team1_stats[agent2_id]
    team2_stats = match.team2_stats

    team1_total = agent1_stats.total_score + agent2_stats.total_score

    print(f"\nTeam 1 (Agents {agent1_id} & {agent2_id}):")
    print(f"  Total Team Score: {team1_total}")
    print(f"  Agent {agent1_id} Score: {agent1_stats.total_score}")
    print(f"  Agent {agent2_id} Score: {agent2_stats.total_score}")

    agent1_coop = agent1_stats.action_a_count / (agent1_stats.action_a_count + agent1_stats.action_b_count) * 100
    agent2_coop = agent2_stats.action_a_count / (agent2_stats.action_a_count + agent2_stats.action_b_count) * 100

    print(f"  Agent {agent1_id} Cooperation Rate: {agent1_coop:.1f}%")
    print(f"  Agent {agent2_id} Cooperation Rate: {agent2_coop:.1f}%")

    print(f"\nTeam 2 (Agent {team2_id}):")
    print(f"  Score: {team2_stats.total_score}")
    team2_coop = team2_stats.action_a_count / (team2_stats.action_a_count + team2_stats.action_b_count) * 100
    print(f"  Cooperation Rate: {team2_coop:.1f}%")

    print(f"\n{'='*70}\n")


def create_tournament_graph():
    workflow = StateGraph(TournamentState)

    workflow.add_node("player1_plan", player1_plan)
    workflow.add_node("player1_move", player1_move)
    workflow.add_node("player2_plan", player2_plan)
    workflow.add_node("player2_move", player2_move)
    workflow.add_node("start_round", start_round)
    workflow.add_node("play_round", play_round)

    workflow.set_entry_point("start_round")

    workflow.add_edge("start_round", "player1_plan")
    workflow.add_edge("player1_plan", "player1_move")
    workflow.add_edge("player1_move", "play_round")
    workflow.add_edge("start_round", "player2_plan")
    workflow.add_edge("player2_plan", "player2_move")
    workflow.add_edge("player2_move", "play_round")

    workflow.add_conditional_edges("play_round", should_continue)
    graph = workflow.compile()

    #    png_data = graph.get_graph().draw_mermaid_png()
    #    with open("plots/langgraph_workflow.png", "wb") as f:
    #        f.write(png_data)

    return graph


def create_tournament_graph_2v1():
    """Create LangGraph workflow for 2-vs-1 team game.

    First-round flow (discussion phase):
        start_round → team_discussion → adversary_discussion → fan-out to planning nodes

    Subsequent rounds (no discussion):
        start_round → fan-out to planning nodes directly
    """
    workflow = StateGraph(TournamentState)

    # Add a synchronisation node that all planning branches start from
    def begin_gameplay(_state: TournamentState):
        """No-op sync node: gameplay begins after all discussion phases."""
        return {}

    # Add nodes
    workflow.add_node("start_round", start_round_2v1)
    workflow.add_node("team_discussion", team_discussion_node)
    workflow.add_node("adversary_discussion", adversary_discussion_node)
    workflow.add_node("begin_gameplay", begin_gameplay)
    workflow.add_node("team1_agent1_plan", team1_agent1_plan)
    workflow.add_node("team1_agent1_move", team1_agent1_move)
    workflow.add_node("team1_agent2_plan", team1_agent2_plan)
    workflow.add_node("team1_agent2_move", team1_agent2_move)
    workflow.add_node("team2_plan", team2_plan)
    workflow.add_node("team2_move", team2_move_node)
    workflow.add_node("play_round", play_round_2v1)

    # Entry point
    workflow.set_entry_point("start_round")

    # Conditional: Run discussion phases only on the first round of each match
    def should_discuss(state: TournamentState) -> str:
        match = state.matches[state.current_match_idx]
        if match.current_round == 1 and not match.discussion_history:
            return "team_discussion"
        return "begin_gameplay"

    workflow.add_conditional_edges("start_round", should_discuss, {
        "team_discussion": "team_discussion",
        "begin_gameplay": "begin_gameplay",
    })

    # Sequential discussion: Team 1 discusses → Team 2 strategizes → gameplay
    workflow.add_edge("team_discussion", "adversary_discussion")
    workflow.add_edge("adversary_discussion", "begin_gameplay")

    # Fan out from begin_gameplay to all parallel planning nodes
    workflow.add_edge("begin_gameplay", "team1_agent1_plan")
    workflow.add_edge("begin_gameplay", "team1_agent2_plan")
    workflow.add_edge("begin_gameplay", "team2_plan")

    # Planning → move generation
    workflow.add_edge("team1_agent1_plan", "team1_agent1_move")
    workflow.add_edge("team1_agent2_plan", "team1_agent2_move")
    workflow.add_edge("team2_plan", "team2_move")

    # All moves feed into play_round
    workflow.add_edge("team1_agent1_move", "play_round")
    workflow.add_edge("team1_agent2_move", "play_round")
    workflow.add_edge("team2_move", "play_round")

    # Conditional: Continue or end
    workflow.add_conditional_edges("play_round", should_continue_2v1)

    return workflow.compile()


def should_continue_2v1(state: TournamentState) -> str:
    """Determine if tournament should continue."""
    if state.current_match_idx >= len(state.matches.keys()):
        return END

    current_match = state.matches[state.current_match_idx]

    if not current_match.completed:
        return "start_round"

    return "start_round"


def play_round_2v1(state: TournamentState):
    """Execute a round in 2-vs-1 game."""
    match = state.matches[state.current_match_idx]

    # Get moves
    agent1_id = match.team1_member_ids[0]
    agent2_id = match.team1_member_ids[1]
    team2_id = match.team2_member_id

    agent1_move = match.team1_moves[agent1_id]
    agent2_move = match.team1_moves[agent2_id]
    team2_move = match.team2_move

    # Calculate payoffs from PAYOFFS_2V1
    payoff_key = (agent1_move, agent2_move, team2_move)
    team1_total_score, team2_score = PAYOFFS_2V1[payoff_key]

    # Update round results
    match.round_results.append({
        'team1': {agent1_id: agent1_move, agent2_id: agent2_move},
        'team2': team2_move
    })
    match.round_scores.append((team1_total_score, team2_score))

    # Track first interaction (for research metrics)
    if match.is_first_interaction:
        # Record cooperation for Team 1 members
        for agent_id in match.team1_member_ids:
            move = match.team1_moves[agent_id]
            cooperated = (move == "action_a")
            state.first_interaction_coop_intragroup.append(cooperated)

        # Record Team 2 cooperation
        team2_cooperated = (team2_move == "action_a")
        state.first_interaction_coop_intergroup.append(team2_cooperated)

        match.is_first_interaction = False

    # Update stats - Team 1 gets collective score (distributed equally)
    agent1_share = int(team1_total_score / 2)
    agent2_share = int(team1_total_score / 2)

    match.team1_stats[agent1_id].total_score += agent1_share
    match.team1_stats[agent2_id].total_score += agent2_share
    match.team2_stats.total_score += team2_score

    # Update action counts
    if agent1_move == "action_a":
        match.team1_stats[agent1_id].action_a_count += 1
    else:
        match.team1_stats[agent1_id].action_b_count += 1

    if agent2_move == "action_a":
        match.team1_stats[agent2_id].action_a_count += 1
    else:
        match.team1_stats[agent2_id].action_b_count += 1

    if team2_move == "action_a":
        match.team2_stats.action_a_count += 1
    else:
        match.team2_stats.action_b_count += 1

    # Update move histories
    match.team1_stats[agent1_id].move_history.append(agent1_move)
    match.team1_stats[agent2_id].move_history.append(agent2_move)
    match.team2_stats.move_history.append(team2_move)

    # Update score histories
    match.team1_stats[agent1_id].score_history.append(match.team1_stats[agent1_id].total_score)
    match.team1_stats[agent2_id].score_history.append(match.team1_stats[agent2_id].total_score)
    match.team2_stats.score_history.append(match.team2_stats.total_score)

    # Update cooperation rate histories
    for agent_id in [agent1_id, agent2_id]:
        stats = match.team1_stats[agent_id]
        total_moves = stats.action_a_count + stats.action_b_count
        coop_rate = (stats.action_a_count / total_moves) * 100 if total_moves > 0 else 0
        stats.action_a_rate_history.append(coop_rate)

    team2_total_moves = match.team2_stats.action_a_count + match.team2_stats.action_b_count
    team2_coop_rate = (match.team2_stats.action_a_count / team2_total_moves) * 100 if team2_total_moves > 0 else 0
    match.team2_stats.action_a_rate_history.append(team2_coop_rate)

    # Increment round counter
    match.current_round += 1
    state.round_number += 1

    # Check if match should end
    team1_total_rounds = max(
        state.players[agent1_id].action_a_count + state.players[agent1_id].action_b_count +
        match.team1_stats[agent1_id].action_a_count + match.team1_stats[agent1_id].action_b_count,
        state.players[agent2_id].action_a_count + state.players[agent2_id].action_b_count +
        match.team1_stats[agent2_id].action_a_count + match.team1_stats[agent2_id].action_b_count
    )

    team2_total_rounds = (state.players[team2_id].action_a_count +
                         state.players[team2_id].action_b_count +
                         match.team2_stats.action_a_count +
                         match.team2_stats.action_b_count)

    # Check keep_playing flags (match ends if ANY member wants to quit)
    team1_wants_continue = all(match.team1_keep_playing.values())

    end_match = (
        (not team1_wants_continue or not match.team2_keep_playing) or
        match.current_round > MAX_ROUNDS_PER_MATCH or
        team1_total_rounds >= MAX_TOTAL_ROUNDS or
        team2_total_rounds >= MAX_TOTAL_ROUNDS
    )

    old_match_idx = state.current_match_idx
    if end_match:
        match.completed = True

        # Update player stats in tournament state
        for agent_id in match.team1_member_ids:
            state.update_player_stats(agent_id, match.team1_stats[agent_id], match.team2_stats)
        state.update_player_stats(team2_id, match.team2_stats, match.team1_stats[agent1_id])

        # Update behavioral metrics (SFEM/traits)
        for agent_id in match.team1_member_ids:
            state.update_nth_stats(agent_id, match.team1_stats[agent_id], match.team2_stats)
        state.update_nth_stats(team2_id, match.team2_stats, match.team1_stats[agent1_id])

        # Print match results
        print_match_results_2v1(match)

        # Move to next match
        state.current_match_idx += 1

        if state.current_match_idx < len(state.matches.keys()):
            # Initialize next match with carried-over plans
            new_match = state.matches[state.current_match_idx]
            for agent_id in new_match.team1_member_ids:
                new_match.team1_stats[agent_id].plan = state.players[agent_id].plan
                new_match.team1_stats[agent_id].critique = state.players[agent_id].critique
            new_match.team2_stats.plan = state.players[new_match.team2_member_id].plan
            new_match.team2_stats.critique = state.players[new_match.team2_member_id].critique
            state.matches[state.current_match_idx] = new_match

        print(f"End of match {state.current_match_idx} out of {len(state.matches)}")

    return {
        "matches": {old_match_idx: match},
        "current_match_idx": state.current_match_idx,
        "round_number": state.round_number,
        "first_interaction_coop_intragroup": state.first_interaction_coop_intragroup,
        "first_interaction_coop_intergroup": state.first_interaction_coop_intergroup,
    }


# %% Run simulation


def run_tournament(replication, model):
    print(f"Starting Tournament with {NUM_GROUPS} groups of {GROUP_SIZE} players each...")

    # Initialize players and groups
    players, groups = initialize_groups_and_players()
    initial_state = TournamentState(
        players=players,
        groups=groups,
        matches={},
        current_match_idx=0,
        round_number=0,
        intergroup_competition_results=[],
        experiment_condition=current_tournament_condition
    )

    # Generate matchups
    all_players = list(range(NUM_GROUPS * GROUP_SIZE))
    # Condition 1: Only repeated interactions (within-group matches)
    if current_tournament_condition == "repeated_only":
        for i, (p1, p2) in enumerate(combinations(all_players, 2)):
            initial_state.matches[i] = MatchState(player1_id=p1, player2_id=p2)

    # Condition 2: Only intergroup competition (between-group matches)
    elif current_tournament_condition == "competition_only":
        for g1, g2 in combinations(range(NUM_GROUPS), 2):
            g1_members = [p for p in all_players if initial_state.players[p].group_id == g1]
            g2_members = [p for p in all_players if initial_state.players[p].group_id == g2]
            # All player of g1 play against all players of g2
            for i, (p1, p2) in enumerate(product(g1_members, g2_members)):
                initial_state.matches[i] = MatchState(player1_id=p1, player2_id=p2)

    # Condition 3: Super-additive (both mechanisms)
    else:  # "super_additive"
        # All possible matches, simplified if necessary
        all_edges = list(combinations(all_players, 2))
        if SIMPLIFY_MATCHES_GRAPH:
            all_edges = random.choices(all_edges, k=MAX_MATCHES_NUM)
        for i, (p1, p2) in enumerate(all_edges):
            initial_state.matches[i] = MatchState(player1_id=p1, player2_id=p2)

    graph = create_tournament_graph()
    final_state = graph.invoke(initial_state, {"recursion_limit": 10000})

    # Display final results
    print("\n" + "=" * 50)
    print("TOURNAMENT RESULTS")
    print("=" * 50)

    sorted_players = sorted(final_state.get('players').items(), key=lambda x: x[1].total_score, reverse=True)
    for player_id, stats in sorted_players:
        total_moves = stats.action_a_count + stats.action_b_count
        action_a_rate = (stats.action_a_count / total_moves * 100) if total_moves > 0 else 0
        print(f"Player {player_id}: Score={stats.total_score}, "
              f"Action A Rate={action_a_rate:.1f}% ({stats.action_a_count}/{total_moves})")

    print(f"\nTotal matches played: {len(final_state.get('matches'))}")
    winner = sorted_players[0][0]
    print(f"Tournament Winner: Player {winner} with {sorted_players[0][1].total_score} points!")

    filename = f"results_{current_tournament_condition}_{replication}_{model}.json"

    with open(filename, 'w') as f:
        json.dump(final_state, f, indent=2, default=custom_json_encoder)

    print(f"Replication results saved to {filename}")

    return filename


def run_full_experiment(condition, replications, model):
    # Run the three experimental conditions
    res_files = []
    global current_tournament_condition
    current_tournament_condition = condition
    for replication in range(replications):
        print(f"\n\nRunning experiment: {current_tournament_condition}, replication number {replication}\n")
        res_files.append(run_tournament(replication, model))
    return res_files


# %% 2-vs-1 Tournament Execution

def run_tournament_2v1(replication, model):
    """Run a 2-vs-1 tournament."""
    print(f"Starting 2-vs-1 Tournament...")
    print(f"Team 1: 2 agents | Team 2: 1 agent")
    print(f"Discussion turns: {DISCUSSION_TURNS}\n")

    # Initialize 3 players total (2 for Team 1, 1 for Team 2)
    players = {
        0: PlayerStats(group_id=0, is_team_member=True, team_id=1, teammate_ids=[1]),
        1: PlayerStats(group_id=0, is_team_member=True, team_id=1, teammate_ids=[0]),
        2: PlayerStats(group_id=1, is_team_member=True, team_id=2, teammate_ids=[])
    }

    # Create groups (just for compatibility)
    groups = {
        0: GroupStats(group_id=0, members=[0, 1]),  # Team 1
        1: GroupStats(group_id=1, members=[2])      # Team 2
    }

    initial_state = TournamentState(
        players=players,
        groups=groups,
        matches={},
        current_match_idx=0,
        round_number=0,
        intergroup_competition_results=[],
        experiment_condition="2vs1_team_game"
    )

    # Create single match: Team 1 vs Team 2
    match = MatchState(
        is_team_game=True,
        team1_member_ids=[0, 1],
        team2_member_id=2,
        team1_stats={
            0: PlayerStats(group_id=0, is_team_member=True, team_id=1, teammate_ids=[1]),
            1: PlayerStats(group_id=0, is_team_member=True, team_id=1, teammate_ids=[0])
        },
        team2_stats=PlayerStats(group_id=1, is_team_member=True, team_id=2),
        team1_moves={},
        team1_keep_playing={0: True, 1: True},
        team2_keep_playing=True
    )

    initial_state.matches[0] = match

    # Create and run graph
    graph = create_tournament_graph_2v1()
    final_state = graph.invoke(initial_state, {"recursion_limit": 10000})

    # Display final results
    print("\n" + "=" * 70)
    print("TOURNAMENT RESULTS")
    print("=" * 70)

    agent0 = final_state.get('players')[0]
    agent1 = final_state.get('players')[1]
    agent2 = final_state.get('players')[2]

    team1_score = agent0.total_score + agent1.total_score
    team2_score = agent2.total_score

    print(f"\nTeam 1 Total Score: {team1_score}")
    print(f"  Agent 0: {agent0.total_score}")
    print(f"  Agent 1: {agent1.total_score}")

    print(f"\nTeam 2 Total Score: {team2_score}")
    print(f"  Agent 2: {agent2.total_score}")

    winner = "Team 1" if team1_score > team2_score else "Team 2" if team2_score > team1_score else "Tie"
    print(f"\nWinner: {winner}")

    # Save results
    filename = f"results_2v1_{replication}_{model}.json"
    with open(filename, 'w') as f:
        json.dump(final_state, f, indent=2, default=custom_json_encoder)

    print(f"\nResults saved to {filename}")

    return filename
