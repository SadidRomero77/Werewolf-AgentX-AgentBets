# Copyright 2024 Google LLC
# Modifications for AgentBeats integration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Game Orchestrator - Bridges A2A protocol with existing GameMaster."""

import asyncio
import logging
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime

from green_agent.models import (
    ActionRequest,
    ActionResponse,
    ActionType,
    AssessmentConfig,
    AssessmentResult,
    GamePhase,
    GameState,
    PlayerScore,
    RoleAssignment,
    RoleType,
    TaskUpdate,
)
from green_agent.a2a_client import A2AClient, verify_agent_connectivity
from green_agent.scoring import ScoringEngine, PlayerMetrics, detect_sabotage

logger = logging.getLogger(__name__)


# Role distribution based on player count
ROLE_DISTRIBUTIONS = {
    5: {"werewolves": 1, "seer": 1, "doctor": 1, "villagers": 2},
    6: {"werewolves": 1, "seer": 1, "doctor": 1, "villagers": 3},
    7: {"werewolves": 2, "seer": 1, "doctor": 1, "villagers": 3},
    8: {"werewolves": 2, "seer": 1, "doctor": 1, "villagers": 4},
}

# Game rules template
GAME_RULES = """You are playing Werewolf, a social deduction game.

PLAYERS: {num_players} players
ROLES: {role_counts}

GAME FLOW:
1. NIGHT PHASE: 
   - Werewolves secretly choose a player to eliminate
   - Seer investigates one player's identity
   - Doctor protects one player from elimination

2. DAY PHASE:
   - Players discuss and debate
   - Players vote to eliminate a suspected werewolf

WINNING:
- Villagers win by eliminating ALL werewolves
- Werewolves win when they equal or outnumber villagers

YOUR ROLE: {role}
{role_specific_instructions}
"""

ROLE_INSTRUCTIONS = {
    RoleType.WEREWOLF: """
As a WEREWOLF, you must:
- Deceive villagers about your identity
- Coordinate with fellow werewolves (if any)
- Vote strategically to eliminate villagers
- Avoid detection while casting suspicion on others
- During night, choose wisely who to eliminate (avoid Seer/Doctor if suspected)
""",
    RoleType.VILLAGER: """
As a VILLAGER, you must:
- Analyze player behavior for suspicious patterns
- Identify inconsistencies in claims and accusations
- Build alliances with trusted players
- Vote to eliminate suspected werewolves
- Remember: werewolves will try to deceive you!
""",
    RoleType.SEER: """
As the SEER, you have a special power:
- Each night, you learn ONE player's true role
- Use this information wisely - revealing it makes you a target
- Balance between sharing information and protecting yourself
- Your investigations can save the village!
- Consider the timing of when to reveal what you know
""",
    RoleType.DOCTOR: """
As the DOCTOR, you have a special power:
- Each night, you protect ONE player from elimination
- You can protect yourself
- Try to predict who the werewolves will target
- Protecting the Seer (if you know who they are) is valuable
- Your protection can save key players!
""",
}

# Context templates for each action type - provides rich instructions to purple agents
ACTION_CONTEXTS = {
    ActionType.ELIMINATE: """WEREWOLF NIGHT ACTION: Choose a player to eliminate.

STRATEGIC CONSIDERATIONS:
- Target players who seem most dangerous to your team
- Consider eliminating suspected Seer or Doctor first
- Avoid targeting players who might be protected by the Doctor
- If you've been accused, eliminating your accuser may help

Your fellow werewolves: {teammates}
Remember: Only one elimination happens per night. Choose wisely.""",

    ActionType.PROTECT: """DOCTOR NIGHT ACTION: Choose a player to protect from elimination.

STRATEGIC CONSIDERATIONS:
- You can protect yourself if you feel targeted
- Try to predict who the werewolves will attack
- Protect players who seem valuable (Seer, influential villagers)
- Consider protecting vocal players who might be targets
- Watch for patterns in who werewolves have targeted before

Your protection will save this player if werewolves attack them tonight.""",

    ActionType.INVESTIGATE: """SEER NIGHT ACTION: Choose a player to investigate.

STRATEGIC CONSIDERATIONS:
- Investigate players you're unsure about
- Avoid wasting investigations on obvious villagers
- Consider investigating players who are behaving suspiciously
- Your information is valuable - use it to guide village votes
- Balance revealing info vs. protecting your identity

You will learn if the player is a WEREWOLF or NOT A WEREWOLF.""",

    ActionType.DEBATE: """DAY PHASE: Share your thoughts with the village.

YOU SHOULD:
- Analyze recent events and share observations
- Accuse suspicious players (with reasoning)
- Defend yourself if accused (without being too defensive)
- Build alliances with players you trust
- Watch for inconsistencies in others' statements

If you are a WEREWOLF: Blend in, cast suspicion on villagers, defend fellow wolves subtly.
If you are a VILLAGER/SEER/DOCTOR: Try to identify werewolves, share useful observations.""",

    ActionType.VOTE: """VOTING PHASE: Vote to eliminate a suspected werewolf.

STRATEGIC CONSIDERATIONS:
- Review the debate for suspicious behavior
- Consider who has made inconsistent statements
- Think about who benefits from certain eliminations
- Don't just follow the crowd - use your own judgment
- If you have information (Seer), use it wisely

If you are a WEREWOLF: Vote strategically to avoid suspicion while eliminating villagers.
If you are a VILLAGER: Vote for who you genuinely believe is a werewolf.""",
}


class GameOrchestrator:
    """Orchestrates a Werewolf game via A2A protocol."""

    def __init__(
        self,
        task_id: str,
        participants: Dict[str, str],  # player_name -> endpoint
        config: AssessmentConfig,
        on_task_update: Optional[Callable[[TaskUpdate], None]] = None,
        event_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ):
        self.task_id = task_id
        self.participants = participants
        self.config = config
        self.on_task_update = on_task_update
        self.event_callback = event_callback
        
        # Game state
        self.player_names = list(participants.keys())
        self.num_players = len(self.player_names)
        self.roles: Dict[str, RoleType] = {}
        self.teams: Dict[str, str] = {}  # player -> "werewolves" or "villagers"
        self.alive_players: List[str] = []
        self.eliminated_players: List[str] = []
        
        # Round tracking
        self.current_round = 0
        self.current_phase = GamePhase.NIGHT
        self.debate_history: List[Dict[str, str]] = []
        self.announcements: List[str] = []
        self.observations: Dict[str, List[str]] = {}  # player -> their observations
        
        # Game log for results
        self.game_log: List[Dict[str, Any]] = []
        
        # Scoring
        self.scoring = ScoringEngine()
        self.metrics: Dict[str, PlayerMetrics] = {}
        
        # Client for A2A communication
        self.client: Optional[A2AClient] = None
        
        # Winner
        self.winner: Optional[str] = None

    def _emit_update(self, message: str, details: Optional[Dict] = None):
        """Emit a task update for tracking."""
        update = TaskUpdate(
            task_id=self.task_id,
            message=message,
            round=self.current_round,
            phase=self.current_phase,
            details=details,
        )
        logger.info(f"[Round {self.current_round}] {message}")
        if self.on_task_update:
            self.on_task_update(update)

    async def _emit_event(self, event_type: str, payload: Dict[str, Any]):
        """Emit a real-time event via the callback."""
        if self.event_callback:
            full_event = {
                "type": event_type,
                "timestamp": datetime.now().isoformat(),
                "payload": payload
            }
            # If the callback is async (which it is for ws_manager.broadcast), await it
            if asyncio.iscoroutinefunction(self.event_callback):
                await self.event_callback(full_event)
            else:
                self.event_callback(full_event)

    def _assign_roles(self) -> Dict[str, RoleType]:
        """Randomly assign roles to players."""
        if self.num_players not in ROLE_DISTRIBUTIONS:
            raise ValueError(f"Unsupported player count: {self.num_players}")
        
        distribution = ROLE_DISTRIBUTIONS[self.num_players]
        shuffled = self.player_names.copy()
        random.shuffle(shuffled)
        
        roles = {}
        idx = 0
        
        # Assign werewolves
        for _ in range(distribution["werewolves"]):
            roles[shuffled[idx]] = RoleType.WEREWOLF
            self.teams[shuffled[idx]] = "werewolves"
            idx += 1
        
        # Assign seer
        roles[shuffled[idx]] = RoleType.SEER
        self.teams[shuffled[idx]] = "villagers"
        idx += 1
        
        # Assign doctor
        roles[shuffled[idx]] = RoleType.DOCTOR
        self.teams[shuffled[idx]] = "villagers"
        idx += 1
        
        # Assign villagers
        for _ in range(distribution["villagers"]):
            roles[shuffled[idx]] = RoleType.VILLAGER
            self.teams[shuffled[idx]] = "villagers"
            idx += 1
        
        return roles

    def _get_teammates(self, player_name: str) -> Optional[List[str]]:
        """Get teammates for a player (only for werewolves)."""
        if self.roles.get(player_name) != RoleType.WEREWOLF:
            return None
        return [
            name for name, role in self.roles.items()
            if role == RoleType.WEREWOLF and name != player_name
        ]

    def _build_game_state(self, for_player: str) -> GameState:
        """Build current game state from a player's perspective."""
        return GameState(
            round=self.current_round,
            phase=self.current_phase,
            alive_players=self.alive_players.copy(),
            eliminated_players=self.eliminated_players.copy(),
            debate_so_far=self.debate_history.copy(),
            announcements=self.announcements.copy(),
            your_observations=self.observations.get(for_player, []).copy(),
        )

    async def _send_role_assignments(self):
        """Send role assignments to all players.

        This is where the green agent communicates all game rules and
        role-specific instructions to each purple agent player.
        """
        self._emit_update("Assigning roles to players")

        # Build role counts string for game rules
        distribution = ROLE_DISTRIBUTIONS[self.num_players]
        role_counts = ", ".join([
            f"{count} {role_name}" for role_name, count in distribution.items()
        ])

        tasks = []
        for player_name in self.player_names:
            role = self.roles[player_name]

            # Build complete game rules (without role-specific instructions)
            # This gives the player an overview of the game
            game_rules = GAME_RULES.format(
                num_players=self.num_players,
                role_counts=role_counts,
                role=role.value,
                role_specific_instructions="(See role_description for your specific instructions)",
            )

            # Build role-specific description with full instructions
            # This tells the player exactly how to play their role
            role_description = ROLE_INSTRUCTIONS[role]

            assignment = RoleAssignment(
                task_id=self.task_id,
                player_name=player_name,
                role=role,
                role_description=role_description,
                game_rules=game_rules,
                teammates=self._get_teammates(player_name),
            )
            endpoint = self.participants[player_name]
            tasks.append(self.client.assign_role(endpoint, assignment))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log role assignments
        role_log = {name: role.value for name, role in self.roles.items()}
        self.game_log.append({
            "event": "role_assignment",
            "roles": role_log,
        })

    async def _request_action(
        self,
        player_name: str,
        action: ActionType,
        options: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> ActionResponse:
        """Request an action from a player."""
        request = ActionRequest(
            task_id=self.task_id,
            player_name=player_name,
            action=action,
            game_state=self._build_game_state(player_name),
            options=options,
            context=context,
        )
        
        endpoint = self.participants[player_name]
        response = await self.client.request_action(endpoint, request)
        
        # Track metrics
        if player_name in self.metrics:
            if action == ActionType.DEBATE:
                self.metrics[player_name].total_debates += 1
            elif action == ActionType.VOTE:
                self.metrics[player_name].total_votes += 1
        
        return response

    async def _run_night_phase(self) -> Optional[str]:
        """Run the night phase. Returns eliminated player name or None."""
        self.current_phase = GamePhase.NIGHT
        self._emit_update("Night falls. Special roles take action.")
        
        await self._emit_event("phase_change", {
            "phase": "night", 
            "round": self.current_round
        })
        
        # Get werewolves and their targets
        werewolves = [p for p in self.alive_players if self.roles[p] == RoleType.WEREWOLF]
        potential_targets = [p for p in self.alive_players if p not in werewolves]
        
        eliminated_target = None
        protected_target = None
        
        # Werewolf elimination choice (first werewolf decides)
        if werewolves and potential_targets:
            wolf = werewolves[0]  # Lead wolf decides
            response = await self._request_action(
                wolf,
                ActionType.ELIMINATE,
                options=potential_targets,
                context=ACTION_CONTEXTS[ActionType.ELIMINATE].format(
                    teammates=", ".join(werewolves) if len(werewolves) > 1 else "You are the only werewolf"
                ),
            )
            eliminated_target = response.decision if response.decision in potential_targets else None
            
            if eliminated_target:
                self._emit_update(
                    f"Werewolves target {eliminated_target}",
                    {"werewolf_action": "eliminate", "target": eliminated_target}
                )
                # Add observation for all werewolves
                for wolf_name in werewolves:
                    if wolf_name not in self.observations:
                        self.observations[wolf_name] = []
                    self.observations[wolf_name].append(
                        f"Night {self.current_round}: We targeted {eliminated_target}."
                    )
        
        # Doctor protection choice
        doctors = [p for p in self.alive_players if self.roles[p] == RoleType.DOCTOR]
        if doctors:
            doctor = doctors[0]
            response = await self._request_action(
                doctor,
                ActionType.PROTECT,
                options=self.alive_players,
                context=ACTION_CONTEXTS[ActionType.PROTECT],
            )
            protected_target = response.decision if response.decision in self.alive_players else None
            
            if protected_target:
                self._emit_update(
                    f"Doctor protects {protected_target}",
                    {"doctor_action": "protect", "target": protected_target}
                )
                if doctor not in self.observations:
                    self.observations[doctor] = []
                self.observations[doctor].append(
                    f"Night {self.current_round}: I protected {protected_target}."
                )
                
                # Track doctor metrics
                if doctor in self.metrics:
                    self.metrics[doctor].protections_total += 1
                    if protected_target == eliminated_target:
                        self.metrics[doctor].protections_successful += 1
        
        # Seer investigation
        seers = [p for p in self.alive_players if self.roles[p] == RoleType.SEER]
        if seers:
            seer = seers[0]
            investigation_targets = [p for p in self.alive_players if p != seer]
            if investigation_targets:
                response = await self._request_action(
                    seer,
                    ActionType.INVESTIGATE,
                    options=investigation_targets,
                    context=ACTION_CONTEXTS[ActionType.INVESTIGATE],
                )
                investigated = response.decision if response.decision in investigation_targets else None
                
                if investigated:
                    role = self.roles[investigated]
                    is_werewolf = role == RoleType.WEREWOLF
                    result = "WEREWOLF" if is_werewolf else "NOT A WEREWOLF"
                    
                    self._emit_update(
                        f"Seer investigates {investigated}",
                        {"seer_action": "investigate", "target": investigated}
                    )
                    
                    if seer not in self.observations:
                        self.observations[seer] = []
                    self.observations[seer].append(
                        f"Night {self.current_round}: I investigated {investigated}. Result: {result}."
                    )
                    
                    # Track seer metrics
                    if seer in self.metrics:
                        self.metrics[seer].investigations_total += 1
                        if is_werewolf:
                            self.metrics[seer].investigations_correct += 1
        
        # Resolve night - elimination happens if not protected
        actual_eliminated = None
        if eliminated_target and eliminated_target != protected_target:
            actual_eliminated = eliminated_target
            self.alive_players.remove(actual_eliminated)
            self.eliminated_players.append(actual_eliminated)
            announcement = f"During the night, {actual_eliminated} was eliminated by the werewolves."

            await self._emit_event("player_eliminated", {
                "player_id": actual_eliminated,
                "role": self.roles[actual_eliminated].value,
                "phase": "night",
                "round": self.current_round
            })
            
            # Track werewolf metrics
            for wolf in werewolves:
                if wolf in self.metrics:
                    self.metrics[wolf].eliminations_successful += 1
        else:
            announcement = "The night passes. No one was eliminated (the Doctor saved someone!)."
        
        self.announcements.append(announcement)
        self._emit_update(announcement)
        
        self.game_log.append({
            "event": "night_phase",
            "round": self.current_round,
            "target": eliminated_target,
            "protected": protected_target,
            "eliminated": actual_eliminated,
        })
        
        return actual_eliminated

    async def _run_debate(self) -> List[Dict[str, str]]:
        """Run the debate phase. Returns list of debate entries."""
        self.debate_history = []
        
        # Each player gets to speak once (simplified debate)
        speakers = self.alive_players.copy()
        random.shuffle(speakers)
        
        for speaker in speakers[:min(5, len(speakers))]:  # Max 5 debate turns
            response = await self._request_action(
                speaker,
                ActionType.DEBATE,
                context=ACTION_CONTEXTS[ActionType.DEBATE],
            )
            
            statement = response.decision or "(said nothing)"
            self.debate_history.append({"speaker": speaker, "message": statement})
            
            self._emit_update(
                f"{speaker} speaks",
                {"speaker": speaker, "statement": statement[:100]}
            )
            
            await self._emit_event("player_speak", {
                "player_id": speaker,
                "content": statement,
                "round": self.current_round
            })
        
        return self.debate_history

    async def _run_voting(self) -> Optional[str]:
        """Run voting phase. Returns exiled player or None."""
        self.current_phase = GamePhase.VOTING
        self._emit_update("Voting begins")
        
        votes: Dict[str, str] = {}
        
        # Each alive player votes
        votable = self.alive_players.copy()
        
        for voter in self.alive_players:
            options = [p for p in votable if p != voter]
            response = await self._request_action(
                voter,
                ActionType.VOTE,
                options=options,
                context=ACTION_CONTEXTS[ActionType.VOTE],
            )
            
            voted_for = response.decision if response.decision in options else None
            if voted_for:
                votes[voter] = voted_for
                
                # Track vote accuracy
                if voter in self.metrics:
                    target_team = self.teams.get(voted_for, "")
                    voter_team = self.teams.get(voter, "")
                    
                    if voter_team == "villagers" and target_team == "werewolves":
                        self.metrics[voter].correct_votes += 1
                    elif voter_team == target_team:
                        self.metrics[voter].wrong_votes += 1
                        if detect_sabotage(voter, voter_team, "vote", voted_for, target_team):
                            self.metrics[voter].sabotage_actions += 1
                
                # Track times voted against
                if voted_for in self.metrics:
                    self.metrics[voted_for].times_voted_against += 1
        
        # Count votes
        vote_counts: Dict[str, int] = {}
        for voter, target in votes.items():
            vote_counts[target] = vote_counts.get(target, 0) + 1
        
        # Find majority
        if vote_counts:
            max_votes = max(vote_counts.values())
            majority_threshold = len(self.alive_players) // 2 + 1
            
            if max_votes >= majority_threshold:
                # Find who got the most votes
                candidates = [p for p, v in vote_counts.items() if v == max_votes]
                exiled = random.choice(candidates)  # Tiebreaker
                
                self.alive_players.remove(exiled)
                self.eliminated_players.append(exiled)
                
                announcement = f"The village votes to exile {exiled}. They were a {self.roles[exiled].value}."
                self.announcements.append(announcement)
                self._emit_update(announcement, {"exiled": exiled, "role": self.roles[exiled].value})
                
                await self._emit_event("player_eliminated", {
                    "player_id": exiled,
                    "role": self.roles[exiled].value,
                    "phase": "day",
                    "round": self.current_round
                })
                
                self.game_log.append({
                    "event": "vote_exile",
                    "round": self.current_round,
                    "votes": votes,
                    "exiled": exiled,
                    "role": self.roles[exiled].value,
                })
                
                return exiled
        
        announcement = "No majority reached. No one is exiled."
        self.announcements.append(announcement)
        self._emit_update(announcement)
        
        self.game_log.append({
            "event": "vote_no_majority",
            "round": self.current_round,
            "votes": votes,
        })
        
        return None

    async def _run_day_phase(self) -> Optional[str]:
        """Run the day phase (debate + voting). Returns exiled player or None."""
        self.current_phase = GamePhase.DAY
        self._emit_update("Day breaks. Time for discussion.")
        
        await self._emit_event("phase_change", {
            "phase": "day", 
            "round": self.current_round
        })
        
        # Run debate
        await self._run_debate()
        
        # Run voting
        exiled = await self._run_voting()
        
        return exiled

    def _check_winner(self) -> Optional[str]:
        """Check if there's a winner. Returns 'werewolves', 'villagers', or None."""
        alive_werewolves = len([
            p for p in self.alive_players if self.roles[p] == RoleType.WEREWOLF
        ])
        alive_villagers = len(self.alive_players) - alive_werewolves
        
        if alive_werewolves == 0:
            return "villagers"
        if alive_werewolves >= alive_villagers:
            return "werewolves"
        return None

    def _initialize_metrics(self):
        """Initialize player metrics tracking."""
        for player_name in self.player_names:
            role = self.roles[player_name]
            team = self.teams[player_name]
            self.metrics[player_name] = PlayerMetrics(
                player_name=player_name,
                role=role,
                team=team,
            )
            self.observations[player_name] = []

    def _finalize_metrics(self):
        """Finalize metrics when game ends."""
        for player_name, metrics in self.metrics.items():
            metrics.won = (self.winner == metrics.team)
            metrics.survived = (player_name in self.alive_players)
            metrics.rounds_survived = self.current_round

    async def run_game(self) -> AssessmentResult:
        """Run the complete game and return results."""
        start_time = datetime.now()
        
        async with A2AClient(timeout=self.config.timeout_seconds) as client:
            self.client = client
            
            # Verify all agents are reachable
            self._emit_update("Verifying agent connectivity")
            connectivity = await verify_agent_connectivity(
                self.participants,
                timeout=10.0,
            )
            
            unreachable = [p for p, ok in connectivity.items() if not ok]
            if unreachable:
                raise RuntimeError(f"Agents unreachable: {unreachable}")
            
            # Assign roles
            self.roles = self._assign_roles()
            await self._send_role_assignments()
            
            # Initialize game state
            self.alive_players = self.player_names.copy()
            self._initialize_metrics()

            await self._emit_event("game_start", {
                "game_id": self.task_id,
                "players": self.player_names,
                "roles": {n: r.value for n, r in self.roles.items()}
            })
            
            self._emit_update(
                f"Game starting with {self.num_players} players",
                {"players": self.player_names}
            )
            
            # Main game loop
            while not self.winner and self.current_round < self.config.max_rounds:
                self.current_round += 1
                self._emit_update(f"=== Round {self.current_round} begins ===")
                
                # Night phase
                await self._run_night_phase()
                self.winner = self._check_winner()
                if self.winner:
                    break
                
                # Day phase
                await self._run_day_phase()
                self.winner = self._check_winner()
            
            # Set winner if max rounds reached
            if not self.winner:
                # Werewolves win if they survived to max rounds
                alive_werewolves = len([
                    p for p in self.alive_players if self.roles[p] == RoleType.WEREWOLF
                ])
                self.winner = "werewolves" if alive_werewolves > 0 else "villagers"
            
            self.current_phase = GamePhase.ENDED
            self._emit_update(
                f"Game over! {self.winner.upper()} win!",
                {"winner": self.winner, "rounds": self.current_round}
            )
            
            # Finalize metrics and generate scores
            self._finalize_metrics()
            
            scores = []
            for player_name, metrics in self.metrics.items():
                player_score = self.scoring.generate_player_score(
                    metrics=metrics,
                    total_rounds=self.current_round,
                    total_players=self.num_players,
                )
                scores.append(player_score)
            
            # Calculate aggregate metrics
            aggregate = {
                "total_rounds": self.current_round,
                "game_duration_seconds": (datetime.now() - start_time).total_seconds(),
                # "winner": self.winner, # Removed to avoid validation error (expected float)
                "werewolf_survival_rate": len([
                    p for p in self.alive_players if self.roles[p] == RoleType.WEREWOLF
                ]) / len([p for p, r in self.roles.items() if r == RoleType.WEREWOLF]),
            }
            
            await self._emit_event("game_over", {
                "winner": self.winner,
                "rounds": self.current_round
            })
            
            return AssessmentResult(
                task_id=self.task_id,
                winner=self.winner,
                rounds_played=self.current_round,
                game_log=self.game_log,
                scores=scores,
                aggregate_metrics=aggregate,
            )
