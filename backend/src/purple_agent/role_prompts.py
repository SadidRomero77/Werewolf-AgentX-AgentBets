# Copyright 2024 Google LLC
# Modifications for AgentBeats integration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Generic prompts for Purple Agent players.

This module provides game-agnostic prompts that work with any game.
All game-specific rules and instructions come from the Green Agent.
The Purple Agent is a "generic player" that follows instructions.
"""

from typing import Any, Dict, List, Optional


def get_system_prompt(player_name: str, role_info: str) -> str:
    """Generate a generic system prompt using info from the green agent.

    Args:
        player_name: The name assigned to this player
        role_info: Complete role description and game rules from green agent

    Returns:
        System prompt for the LLM
    """
    return f"""You are {player_name}, an AI player participating in a game.

IMPORTANT INSTRUCTIONS:
1. Follow the rules and instructions provided by the game master below
2. Make strategic decisions based on your role and objectives
3. Analyze the game state and other players' behavior
4. Respond in JSON format with your decision and reasoning

YOUR ROLE AND GAME RULES:
{role_info}

RESPONSE FORMAT:
Always respond with valid JSON:
{{
    "decision": "your_choice_here",
    "reasoning": "brief explanation of your decision"
}}
"""


def get_action_prompt(
    action: str,
    context: str,
    options: List[str],
    game_state: Dict[str, Any],
    observations: List[str],
) -> str:
    """Generate an action prompt using info from the green agent.

    Args:
        action: The type of action requested (e.g., "vote", "debate")
        context: Context/instructions from the green agent for this action
        options: Valid choices for this action
        game_state: Current game state from the green agent
        observations: Player's private observations

    Returns:
        Action prompt for the LLM
    """
    # Format game state
    round_num = game_state.get("round", "?")
    phase = game_state.get("phase", "unknown")
    alive = game_state.get("alive_players", [])
    eliminated = game_state.get("eliminated_players", [])
    announcements = game_state.get("announcements", [])
    debate = game_state.get("debate_so_far", [])

    # Format debate history
    debate_text = ""
    if debate:
        for entry in debate:
            speaker = entry.get("speaker", "?")
            message = entry.get("message", "")
            debate_text += f"  - {speaker}: {message}\n"
    else:
        debate_text = "  (No debate yet)\n"

    # Format observations
    obs_text = ""
    if observations:
        for obs in observations:
            obs_text += f"  - {obs}\n"
    else:
        obs_text = "  (None)\n"

    # Format announcements
    announce_text = ""
    if announcements:
        for ann in announcements:
            announce_text += f"  - {ann}\n"
    else:
        announce_text = "  (None)\n"

    # Format options
    options_text = ", ".join(options) if options else "(no specific options)"

    return f"""ACTION REQUIRED: {action.upper()}

INSTRUCTIONS FROM GAME MASTER:
{context}

CURRENT GAME STATE:
- Round: {round_num}
- Phase: {phase}
- Alive players: {', '.join(alive) if alive else '(unknown)'}
- Eliminated players: {', '.join(eliminated) if eliminated else '(none)'}

RECENT ANNOUNCEMENTS:
{announce_text}
DEBATE SO FAR:
{debate_text}
YOUR PRIVATE OBSERVATIONS:
{obs_text}
AVAILABLE OPTIONS: {options_text}

Make your decision based on the game rules, your role, and the current situation.
Respond with JSON: {{"decision": "your_choice", "reasoning": "why"}}
"""
