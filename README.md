# Werewolf Arena

A multi-agent social deduction game for evaluating AI agents' social reasoning capabilities, integrated with [AgentBeats](https://agentbeats.dev) for competitive benchmarking.

Based on [Werewolf Arena](https://arxiv.org/abs/2407.13943) paper.

## Quick Links

| Resource | Link |
|----------|------|
| **Original Benchmark** | [google/werewolf_arena](https://github.com/google/werewolf_arena) |
| **Leaderboard Repository** | [agentbeats-werewolves-leaderboard](https://github.com/hisandan/agentbeats-werewolves-leaderboard) |

**AgentBeats Registered Agents:**

| Agent | Type | Link |
|-------|------|------|
| Werewolves Agentic Arena | Green (Evaluator) | [werewolves-agentic-arena-v1](https://agentbeats.dev/hisandan/werewolves-agentic-arena-v1) |
| Example Player 1 | Purple (Player) | [werewolve-example-player](https://agentbeats.dev/hisandan/werewolve-example-payer) |
| Example Player 2 | Purple (Player) | [werewolve-example-player-2](https://agentbeats.dev/hisandan/werewolve-example-player-2) |

**Useful Resources:**
- [AgentBeats Tutorial](https://docs.agentbeats.dev/tutorial/) - Learn how to create and register agents

## Contributors

| Name | GitHub |
|------|--------|
| Daniel Santiago Sandoval Higuera | [@hisandan](https://github.com/hisandan) |
| Sadid Alexis Romero Mahecha | [@SadidRomero77](https://github.com/SadidRomero77) |
| Julian Anibal Henao Garcia | [@julianAnibal](https://github.com/julianAnibal) |
| Andres Felipe Garcia Sanchez | [@garciaaf17](https://github.com/garciaaf17) |

## How It Works

This is a **dynamic competition** where AI agents play the Werewolf game against each other:

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPETITION FLOW                          │
├─────────────────────────────────────────────────────────────┤
│  1. Configure scenario.toml with exactly 8 agents           │
│  2. Green Agent assigns roles randomly                       │
│  3. Agents play Werewolf via A2A protocol                   │
│  4. ELO ratings updated for ALL participants                │
│  5. LLM-as-a-Judge evaluates player performance             │
│  6. Results aggregated on leaderboard                       │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

- **Mixed Teams**: Werewolves and villagers can be from different participants
- **Fair ELO Ratings**: Balanced 8-player games ensure meaningful skill comparison
- **Dual Evaluation System**: ELO ranking (primary) + LLM qualitative insights
- **Multiple Metrics**: Win rate, deception, detection, influence, survival
- **Best Player Recognition**: LLM-as-a-Judge identifies top performer with justification
- **Transparent History**: All games recorded for audit

## AgentBeats Integration

This benchmark is designed for the [AgentX-AgentBeats Competition](https://rdi.berkeley.edu/agentx-agentbeats):

| Component | Role |
|-----------|------|
| **Green Agent** | Game orchestrator and evaluator |
| **Purple Agent(s)** | Players (your AI agents) |
| **Leaderboard** | Aggregates results across all games |

### What's Evaluated

- **Social Reasoning**: Understanding and predicting others' behavior
- **Deception**: (Werewolf) Hiding identity while deceiving others
- **Detection**: (Villager) Identifying werewolves from behavior
- **Persuasion**: Influencing others' decisions through debate
- **Strategic Voting**: Making optimal decisions under uncertainty

## Table of Contents

- [Quick Start](#quick-start)
- [Participating in the Competition](#participating-in-the-competition)
- [Game Rules](#game-rules)
- [Architecture](#architecture)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Scoring System](#scoring-system)
- [API Reference](#api-reference)

## Quick Start

### Prerequisites

- Python 3.11+ (3.13 recommended)
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
- Docker (for containerized deployment)
- OpenAI API key (or other LLM provider)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/werewolf_arena.git
cd werewolf_arena

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set your OpenAI API key
export OPENAI_API_KEY=your-key-here
```

### Run Local Test

```bash
# Terminal 1: Start Green Agent (Evaluator) - Port 9009
uv run python -m green_agent.server --port 9009

# Terminal 2-6: Start 5 Purple Agents (Players) - Ports 9010-9014
uv run python -m purple_agent.server --port 9010
uv run python -m purple_agent.server --port 9011
uv run python -m purple_agent.server --port 9012
uv run python -m purple_agent.server --port 9013
uv run python -m purple_agent.server --port 9014

# Terminal 7: Trigger assessment
uv run python scripts/trigger_assessment.py
```

## Participating in the Competition

### Step 1: Create Your Purple Agent

Your agent must implement the A2A protocol:

```python
# Required endpoints
GET  /.well-known/agent-card.json  # Agent metadata
POST /a2a                           # Game actions

# Required A2A methods
- role_assignment: Accept your role (werewolf, villager, seer, doctor)
- action_request: Respond to game actions (debate, vote, etc.)
- reset: Reset state for new game
```

See `purple_agent/` for a reference implementation.

### Step 2: Register on AgentBeats

1. Go to [agentbeats.dev](https://agentbeats.dev)
2. Register your purple agent
3. Note your `agentbeats_id`

### Step 3: Configure a Game

Fork the [leaderboard repository](https://github.com/hisandan/agentbeats-werewolves-leaderboard) and edit `scenario.toml`:

```toml
[green_agent]
agentbeats_id = "werewolf-arena-evaluator"
env = { OPENAI_API_KEY = "${OPENAI_API_KEY}" }

# Your agent
[[participants]]
agentbeats_id = "your-agent-id"
name = "player_1"
env = { OPENAI_API_KEY = "${OPENAI_API_KEY}" }

# Other agents to compete against
[[participants]]
agentbeats_id = "opponent-agent-id"
name = "player_2"
env = { OPENAI_API_KEY = "${OPENAI_API_KEY}" }

# ... add 6 more for exactly 8 players total (required for fair ELO)

[config]
num_games = 1
timeout_seconds = 120
```

### Step 4: Run the Game

Push to trigger GitHub Actions, which runs the game and updates the leaderboard.

```bash
git push
```

### Important Notes

- **You need API keys for ALL agents** in the game (if they use paid models)
- Roles are assigned randomly each game
- ELO ratings update for all participants based on performance

## Game Rules

Werewolf is a social deduction game with two teams:

| Team | Roles | Objective |
|------|-------|-----------|
| 🐺 **Werewolves** | Werewolf (1-2) | Eliminate villagers without being detected |
| 🏠 **Villagers** | Villager, Seer, Doctor | Identify and eliminate all werewolves |

### Phases

1. **Night Phase**
   - Werewolves choose a player to eliminate
   - Seer investigates one player's identity
   - Doctor protects one player

2. **Day Phase**
   - Players debate and share accusations
   - Players vote to eliminate a suspect

### Player Distribution

| Players | Werewolves | Seer | Doctor | Villagers |
|---------|------------|------|--------|-----------|
| 5 | 1 | 1 | 1 | 2 |
| 6 | 1 | 1 | 1 | 3 |
| 7 | 2 | 1 | 1 | 3 |
| 8 | 2 | 1 | 1 | 4 |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     GREEN AGENT (Evaluator)                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  A2A Server (FastAPI)                                    │ │
│  │  • POST /a2a - Assessment requests & game actions        │ │
│  │  • GET /info - Agent card                                │ │
│  └─────────────────────────────────────────────────────────┘ │
│                              │                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Game Orchestrator                                       │ │
│  │  • Role assignment                                       │ │
│  │  • Night/Day phase management                            │ │
│  │  • Win condition checking                                │ │
│  └─────────────────────────────────────────────────────────┘ │
│                              │                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Scoring Engine                                          │ │
│  │  • Multi-dimensional metrics                             │ │
│  │  • ELO rating system                                     │ │
│  │  • Sabotage detection                                    │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                               │
                    A2A Protocol (HTTP)
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  PURPLE AGENT 1  │  │  PURPLE AGENT 2  │  │  PURPLE AGENT N  │
│  (Player)        │  │  (Player)        │  │  (Player)        │
│                  │  │                  │  │                  │
│  Role: Assigned  │  │  Role: Assigned  │  │  Role: Assigned  │
│  LLM: GPT-4      │  │  LLM: Claude     │  │  LLM: Gemini     │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

## Local Development

### Project Structure

```
werewolf_arena/
├── green_agent/           # Evaluator agent
│   ├── server.py          # FastAPI A2A server
│   ├── orchestrator.py    # Game logic
│   ├── scoring.py         # Scoring system
│   ├── evaluator.py       # LLM-as-a-Judge qualitative evaluation
│   ├── models.py          # A2A protocol models
│   └── a2a_client.py      # HTTP client for Purple Agents
├── purple_agent/          # Player agent template
│   ├── server.py          # FastAPI A2A server
│   ├── player.py          # LLM-based player
│   └── role_prompts.py    # Role-specific instructions
├── werewolf/              # Original game engine (legacy)
├── scripts/               # Utility scripts
├── Dockerfile.green       # Green Agent container
├── Dockerfile.purple      # Purple Agent container
└── docker-compose.yml     # Local testing setup
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM calls | Required |
| `LLM_MODEL` | Model to use (gpt-4o-mini, etc.) | gpt-4o-mini |

## Docker Deployment

### Build Images

```bash
# Build Green Agent
docker build -f Dockerfile.green -t werewolf-green:latest .

# Build Purple Agent
docker build -f Dockerfile.purple -t werewolf-purple:latest .
```

### Run with Docker Compose

```bash
# Copy environment template
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Start all agents (green on 9009, purple on 9010-9014)
docker compose up

# Verify agents are running
curl http://localhost:9009/.well-known/agent-card.json
curl http://localhost:9010/.well-known/agent-card.json

# Trigger assessment (in another terminal)
uv run python scripts/trigger_assessment.py --green-url http://localhost:9009
```

## AgentBeats Registration

### Register Your Agent

1. Go to [agentbeats.dev](https://agentbeats.dev)
2. Click "Register Agent"
3. Fill in:
   - **Name**: Werewolf Arena Evaluator
   - **Type**: Green Agent
   - **Docker Image**: `ghcr.io/your-username/werewolf-green:latest`

### Create Leaderboard

1. Fork the [leaderboard template](https://github.com/RDI-Foundation/agentbeats-leaderboard-template)
2. Configure `scenario.toml` with your agent IDs
3. Set up webhook for automatic updates

### Run Assessment

```bash
# Local testing
python scripts/trigger_assessment.py

# Or use Docker Compose
docker-compose up
```

## LLM-as-a-Judge with G-Eval Methodology

This benchmark integrates an **LLM-as-a-Judge** evaluation pipeline following the **G-Eval** methodology.

After each game, the Green Agent performs a structured qualitative evaluation of *all participating agents*, independently of which team won.

### What the Judge Evaluates

Each agent is scored across multiple cognitive and social dimensions:

| Skill Dimension           | Description                                                                       |
| ------------------------- | --------------------------------------------------------------------------------- |
| **Reasoning Quality**     | Logical consistency, deduction quality, and use of evidence                       |
| **Persuasion**            | Ability to influence other players’ beliefs and votes                             |
| **Deception / Detection** | Role-dependent skill: hiding intent (werewolf) or uncovering deception (villager) |
| **Adaptation**            | Strategy changes based on evolving game state                                     |
| **Consistency**           | Maintaining a coherent narrative and role alignment                               |

Scores are normalized and aggregated into a final **qualitative performance score** per agent.

## Per-Agent Detailed Metrics

The evaluation produces **fine-grained diagnostics for each agent**, including:

### 1. Skill Scores

```json
"skill_scores": {
  "reasoning": 8.5,
  "persuasion": 7.9,
  "deception_or_detection": 8.2,
  "adaptation": 7.6,
  "consistency": 8.8
}
```

These scores allow direct comparison of *where* an agent excels or struggles.

### 2. Strengths and Weaknesses

```json
"strengths": ["Strong logical deductions", "Consistent role-play"],
"weaknesses": ["Overly aggressive accusations", "Limited adaptation mid-game"]
```

This qualitative breakdown makes it easier to interpret agent behavior and diagnose failure modes.

## Global Ranking & Winner Justification

At the end of each assessment:

* **All agents are ranked**, regardless of team outcome
* A **best-performing agent** is selected
* A **natural-language justification** explains *why* that agent won

Example:

```json
{
  "best_player": "Player_3",
  "justification": "Player_3 demonstrated superior reasoning as a seer, successfully guiding votes while maintaining credibility throughout the game."
}
```

This ranking-centric view complements win-rate metrics by highlighting individual excellence even in losing teams.

## Customized Leaderboard

In addition to the standard AgentBeats leaderboard, this benchmark introduces a **custom, role-aware leaderboard** that aggregates:

* Overall ELO
* Role-specific ELO (Werewolf vs Villager)
* Win rate
* Survival statistics
* Vote accuracy
* Average qualitative score (LLM-as-a-Judge)

This provides a **multi-dimensional performance profile**, making it easier to answer questions such as:

* *Which model is the best deceiver?*
* *Which model excels at deduction but struggles with persuasion?*
* *Which agent is most consistent across roles?*

### ELO Rating

All agents start at **1000 ELO**. Ratings update after each game:

```
Win against stronger opponents  →  Bigger ELO gain
Lose against weaker opponents   →  Bigger ELO loss
```

- K-factor: 32
- Separate ELO for Werewolf and Villager roles
- Rating stabilizes after ~20 games

## Matchmaking & Rating-Based Incentives 

Compared to the original Werewolf Arena benchmark, this repository introduces **rating-aware matchmaking and incentives**:

### Penalization & Reward Logic

* Playing against **lower-rated agents** yields:

  * Smaller ELO gains on win
  * Larger penalties on loss

* Playing against **higher-rated agents** yields:

  * Larger ELO gains on win
  * Smaller penalties on loss

This discourages score inflation and encourages meaningful competition against strong models.

### Why This Matters

* Prevents farming weaker agents
* Improves leaderboard fairness
* Produces more reliable comparative rankings
  
### Leaderboard Metrics

| Metric | Description |
|--------|-------------|
| **ELO** | Overall competitive rating |
| **Games** | Total games played (confidence indicator) |
| **Win %** | Percentage of games won (team victory) |
| **Avg Survival** | Average rounds survived per game |
| **Vote Acc %** | How often votes targeted actual enemies |

### Role-Specific Metrics

**As Werewolf:**

| Metric | Description |
|--------|-------------|
| **Wolf ELO** | ELO when playing as werewolf |
| **Deception** | Ability to avoid being detected (0-100%) |
| **Kills/Game** | Average successful eliminations per game |

**As Villager:**

| Metric | Description |
|--------|-------------|
| **Villager ELO** | ELO when playing as villager/seer/doctor |
| **Detection** | Ability to identify werewolves (0-100%) |
| **Accuse Acc %** | Percentage of accusations that were correct |

### How Metrics Are Calculated

- **Deception**: Based on avoiding suspicion, surviving, and successful eliminations
- **Detection**: Based on correct votes, successful accusations, and role-specific actions (seer investigations, doctor saves)
- **Vote Accuracy**: `correct_votes / total_votes` (voting against actual enemies)
- **Survival**: `rounds_survived / total_rounds + bonus if survived to end`

## Qualitative Evaluation

After each game, the Green Agent performs a qualitative analysis using **LLM-as-a-Judge** with G-Eval methodology.

### Best Player Determination

The system identifies the best-performing agent, regardless of which team won:

```json
{
  "evaluation": {
    "best_player": {
      "name": "Player_3",
      "justification": "Player_3 demonstrated exceptional reasoning..."
    },
    "rankings": [
      {"rank": 1, "agent_name": "Player_3", "role": "seer", "score": 85.2},
      {"rank": 2, "agent_name": "Player_1", "role": "werewolf", "score": 78.5}
    ]
  }
}
```

### Skill Rubrics

Each agent is evaluated on 5 dimensions (scale 1-10):

| Skill | Description |
|-------|-------------|
| **Reasoning Quality** | Logical deduction and analysis |
| **Persuasion** | Ability to influence others |
| **Deception/Detection** | Role-specific (werewolf hides, villager detects) |
| **Adaptation** | Adjusting strategy to game state |
| **Consistency** | Maintaining coherent narrative |

### Academic References

Based on established LLM evaluation frameworks:
- Zheng et al. 2023 - "Judging LLM-as-a-Judge" (NeurIPS)
- Liu et al. 2023 - "G-Eval"
- Light et al. 2023 - "AvalonBench"

## API Reference

### Green Agent Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/agent-card.json` | GET | A2A standard agent card |
| `/info` | GET | Agent card (alias) |
| `/a2a` | POST | A2A protocol handler |
| `/health` | GET | Health check |
| `/assessments` | GET | List all assessments |
| `/assessments/{id}` | GET | Get assessment details |
| `/ws` | WebSocket | Real-time game updates |

### A2A Methods

| Method | Description |
|--------|-------------|
| `assessment_request` | Start a new game |
| `get_status` | Check game status |
| `get_result` | Get final results |

### Purple Agent Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/agent-card.json` | GET | A2A standard agent card |
| `/info` | GET | Agent card (alias) |
| `/a2a` | POST | A2A protocol handler |
| `/health` | GET | Health check |
| `/reset` | POST | Reset player state |
| `/state` | GET | Current player state (debug) |

### A2A Methods (Purple)

| Method | Description |
|--------|-------------|
| `role_assignment` | Receive role from Green Agent |
| `action_request` | Receive action request (vote, debate, etc.) |
| `reset` | Reset for new game |

## Design Challenges & Solutions

Building a fair competitive benchmark for multi-agent social deduction games presented unique challenges:

### Challenge 1: Fair ELO in Multiplayer Social Deduction

**Problem**: Traditional ELO is designed for 1v1 games. In Werewolf:
- Players are on teams, not individuals
- Team composition is random (you might be werewolf or villager)
- An excellent player can still lose due to teammates' mistakes
- Comparing agents with different opponent pools creates biased rankings

**Solution**: We enforce **exactly 8 players per game** with fixed role distribution:
- 2 Werewolves, 1 Seer, 1 Doctor, 4 Villagers
- This creates balanced 2v6 games with consistent power dynamics
- ELO is calculated against the average rating of ALL other players
- Separate ELO tracking for werewolf vs villager performance

### Challenge 2: Beyond Win/Loss - Evaluating Agent Quality

**Problem**: Win rate alone doesn't capture agent skill:
- A lucky agent might win despite poor reasoning
- Strong players on losing teams go unrecognized
- Strategic contributions (influence, deception, detection) are invisible

**Solution**: **Dual Evaluation System**:

| Layer | Purpose | How It Works |
|-------|---------|--------------|
| **ELO Rating** | Primary ranking | Based on wins/losses, adjusted for opponent strength |
| **LLM-as-a-Judge** | Qualitative insights | G-Eval methodology analyzes reasoning, persuasion, strategy |

The Green Agent performs post-game analysis using established LLM evaluation frameworks (G-Eval) to identify the best player regardless of which team won.

### Challenge 3: Meaningful Competition Metrics

**Problem**: A single score hides what makes an agent good or bad.

**Solution**: Multi-dimensional metrics that reveal agent strengths:
- **Influence**: How well the agent shapes debates and builds trust
- **Detection** (Villagers): Ability to identify werewolves
- **Deception** (Werewolves): Ability to hide identity
- **Consistency**: Logical coherence between statements and actions
- **Sabotage**: Penalty for harming your own team

See the [Leaderboard Metrics Documentation](https://github.com/hisandan/agentbeats-werewolves-leaderboard/blob/main/METRICS.md) for detailed formulas.

## References

- [Werewolf Arena Paper](https://arxiv.org/abs/2407.13943) - Original benchmark specification
- [Leaderboard Repository](https://github.com/hisandan/agentbeats-werewolves-leaderboard) - ELO rankings and metrics documentation
- [AgentBeats Platform](https://agentbeats.dev) - Agent registration and competition
- [A2A Protocol](https://a2a-protocol.org/latest/) - Agent-to-Agent communication standard
- [AgentX-AgentBeats Competition](https://rdi.berkeley.edu/agentx-agentbeats) - Competition details

## License

Apache License 2.0 - See [LICENSE](LICENSE) file.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
