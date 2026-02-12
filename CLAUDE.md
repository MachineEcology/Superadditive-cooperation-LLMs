# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements experiments for the paper "Super-additive Cooperation in Language Model Agents" by Filippo Tonini and Lukas Galke. The project simulates tournaments where language model agents, grouped into teams, play iterated Prisoner's Dilemma games to study cooperative behavior under different conditions.

**Paper**: [arXiv:2508.15510](https://arxiv.org/abs/2508.15510)
**Website**: [pippot.github.io/Superadditive-cooperation-LLMs](https://pippot.github.io/Superadditive-cooperation-LLMs/)

## Running Experiments

### Full Experiment Execution
To run the complete experimental pipeline with multiple replications:
```python
python full_IPD_experiment.py
```

This runs all three experimental conditions:
- `repeated_only`: Players engage in repeated interactions within groups
- `competition_only`: Players engage in intergroup competition between groups
- `super_additive`: Both repeated interactions and intergroup competition

### Individual Tournament Execution
To run a single tournament configuration:
```python
from IPDv6 import run_tournament, run_full_experiment

# Run single replication
run_tournament(replication=0, model="qwen3")

# Run multiple replications for one condition
run_full_experiment(condition="super_additive", replications=5, model="qwen3")
```

## Core Architecture

### Game Simulation Framework (IPDv6.py)

The simulation uses **LangGraph** (StateGraph) to orchestrate the tournament workflow. The graph orchestrates:
1. Planning phase (`player1_plan`, `player2_plan`) - Strategic planning with critique iterations
2. Move generation (`player1_move`, `player2_move`) - LLM-based action selection
3. Round execution (`play_round`) - Payoff calculation and state updates

**Key State Management**:
- `TournamentState`: Global tournament state tracking all players, groups, matches, and competition results
- `MatchState`: Individual match state including round-by-round results and player statistics
- `PlayerStats`: Comprehensive player tracking including scores, move history, cooperation rates, plans, and behavioral metrics (SFEM/traits)
- `GroupStats`: Group-level aggregations for intergroup competition

**Critical Configuration Constants** (lines 23-41):
- `PAYOFFS`: Defines the prisoner's dilemma payoff matrix
- `NUM_GROUPS`, `GROUP_SIZE`: Tournament structure
- `MAX_ROUNDS_PER_MATCH`, `MAX_TOTAL_ROUNDS`: Game length limits
- `INTERGROUP_COMPETITION`: Enables/disables group-level competition
- `GROUP_REWARD_MULTIPLIER`: Bonus multiplier for winning group
- `PLANNING_FREQUENCY`: How often agents replan (0 disables planning)
- `CRITIQUE_ITERATIONS`: Number of plan refinement cycles
- `LLM`: The language model used for agent decision-making

### Experimental Conditions

The `current_tournament_condition` global variable controls which experimental condition is active:

1. **repeated_only**: All players can play against each other (within and between groups), but no group rewards
2. **competition_only**: Only between-group matches, with group rewards for the winning team
3. **super_additive**: All matches possible, with group rewards (tests synergy of both mechanisms)

Match generation logic is in `run_tournament()` (lines 959-981).

### LLM-Agent Interaction

**Prompt Templates** (lines 225-315):
- `player_prompt`: Decision-making prompt for choosing actions
- `planner_prompt`: Strategic planning based on game history
- `critic_prompt`: Critical evaluation of plans for improvement
- `meta_prompt`: Comprehension testing (measures agent's understanding of game state)

**Planning System** (lines 516-607):
The agent uses a **plan-critique-replan cycle**:
1. Generate initial plan considering opponent patterns and goals
2. Critique the plan for weaknesses
3. Refine plan based on critique
4. Repeat for `CRITIQUE_ITERATIONS`

This happens every `PLANNING_FREQUENCY` rounds.

### Behavioral Metrics

**SFEM (Strategy Affinity)** in [nicerthanhumansmetrics.py](nicerthanhumansmetrics.py):
Measures how closely player behavior matches known strategies:
- AllC (always cooperate)
- AllD (always defect)
- TFT (tit-for-tat)
- Grim (grim trigger)
- Pavlov (win-stay, lose-shift)

**Trait Metrics**:
- Nice: Never defects first
- Forgiving: Continues cooperating after opponent defection
- Retaliatory: Defects in response to provocation
- Troublemaking: Defects without provocation
- Emulative: Mimics opponent's previous move

These are calculated in `update_nth_stats()` (lines 156-161) after each match.

### Visualization Pipeline

[visualizations.py](visualizations.py) generates comprehensive plots:
- **Cooperation rate evolution**: Shows action_a (cooperation) rates over time with confidence intervals
- **First interaction cooperation**: Tracks one-shot cooperation in initial rounds
- **Intragroup vs Intergroup**: Compares cooperation rates within/between groups
- **Strategy affinities and traits**: Plots behavioral metric evolution across matches
- **Meta-prompt accuracy**: Measures agent comprehension of game state

All visualization functions accept `List[AddableDict]` for multi-replication averaging with confidence intervals.

## Data Storage

Results are saved as JSON files with naming pattern:
```
results_{condition}_{replication}_{model}.json
```

Each file contains the complete `TournamentState` with all player histories, match results, and behavioral metrics.

## Important Implementation Details

### Action Naming Convention
- Internal representation uses `action_a` (cooperate) and `action_b` (defect)
- If `COOPERATE_DEFECT_PROMPT = True`, prompts use natural language "cooperate"/"defect" but are converted back to action_a/action_b in parsing

### LangSmith Tracing
The code includes LangSmith API credentials (lines 16-20) for experiment tracking. These should be moved to environment variables or removed before sharing.

### Error Handling in LLM Parsing
The `generate_next_move()` function (lines 456-513) includes robust fallback logic:
- Attempts structured parsing with Pydantic
- Falls back to regex pattern matching if parsing fails
- Defaults to random action if all parsing attempts fail

### First Interaction Tracking
Special handling for initial encounters between players (lines 205-220, 794-796):
- Distinguishes between intragroup (same group) and intergroup (different groups) first interactions
- Used to measure one-shot cooperation rates

### Group Competition Mechanics
When `INTERGROUP_COMPETITION = True`:
- Groups ranked by total member scores after each match (lines 163-203)
- Winning group receives `GROUP_REWARD_MULTIPLIER` bonus at tournament end
- Competition results logged in `intergroup_competition_results`

## Modifying Experiments

### Changing the Language Model
Update the `LLM` variable (line 41):
```python
from langchain_ollama.llms import OllamaLLM
LLM = OllamaLLM(model="your-model-name")
```

### Adjusting Tournament Structure
Modify these constants at the top of [IPDv6.py](IPDv6.py):
- `NUM_GROUPS`: Number of competing groups
- `GROUP_SIZE`: Players per group
- `MAX_ROUNDS_PER_MATCH`: Maximum rounds in a single match
- `MAX_TOTAL_ROUNDS`: Maximum total rounds per player across all matches

### Customizing Agent Behavior
- Adjust `PLANNING_FREQUENCY`: Higher values = less frequent replanning
- Modify `CRITIQUE_ITERATIONS`: More iterations = more refined strategies (but slower)
- Change payoff matrix in `PAYOFFS` dictionary

## Dependencies

The project uses:
- **LangChain/LangGraph**: Agent orchestration and state management
- **Pydantic**: Structured output parsing and data validation
- **Matplotlib**: Visualization
- **NumPy/SciPy**: Statistical analysis and confidence intervals
- **Ollama**: Local LLM inference

No requirements.txt is present; dependencies must be inferred from imports.
