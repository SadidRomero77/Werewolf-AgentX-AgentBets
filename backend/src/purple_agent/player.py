# Copyright 2024 Google LLC
# Modifications for AgentBeats integration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Purple Agent Player - Generic LLM-based game player.

This is a GENERIC player that works with any game. It receives all game rules
and instructions from the Green Agent and makes decisions based on that info.
No game-specific knowledge is hardcoded here.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from purple_agent.role_prompts import get_system_prompt, get_action_prompt

logger = logging.getLogger(__name__)


class LLMPlayer:
    """Generic LLM-based player for any game.

    This player receives all game rules and role information from the
    Green Agent and makes decisions based on that information.
    No game-specific knowledge is hardcoded.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_retries: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        # Player state (set during role assignment)
        self.player_name: Optional[str] = None
        self.role: Optional[str] = None
        self.system_prompt: Optional[str] = None

        # Game memory
        self.observations: List[str] = []
        self.game_history: List[Dict[str, Any]] = []

        # LLM client
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI-compatible client."""
        if self._client is None:
            # Skip initialization in dummy mode
            if self.model == "dummy":
                return None  # type: ignore
                
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_API_BASE")
            
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable required. "
                    "For OpenRouter, use your OpenRouter API key."
                )
            
            # Support for OpenRouter and other OpenAI-compatible APIs
            if base_url:
                logger.info(f"Using custom API base: {base_url}")
                self._client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self._client = OpenAI(api_key=api_key)
        
        return self._client

    def assign_role(
        self,
        player_name: str,
        role: str,
        role_description: str,
        game_rules: str,
        teammates: Optional[List[str]] = None,
    ):
        """Assign role to this player using info from the green agent.

        Args:
            player_name: Name assigned to this player
            role: The role name (e.g., "werewolf", "villager")
            role_description: Complete role description from green agent
            game_rules: Complete game rules from green agent
            teammates: List of teammate names (if applicable)
        """
        self.player_name = player_name
        self.role = role

        # Build complete role info from green agent's description
        # This is the key change: we use the green agent's info, not hardcoded prompts
        full_role_info = f"{game_rules}\n\n{role_description}"
        if teammates:
            full_role_info += f"\n\nYour teammates: {', '.join(teammates)}"

        # Generate system prompt using info from green agent
        self.system_prompt = get_system_prompt(
            player_name=player_name,
            role_info=full_role_info,
        )

        # Initialize observations (will be populated by game events)
        self.observations = []

        logger.info(f"Player {player_name} assigned role: {role}")

    def reset(self):
        """Reset player state for a new game."""
        self.player_name = None
        self.role = None
        self.system_prompt = None
        self.observations = []
        self.game_history = []

    def add_observation(self, observation: str):
        """Add a private observation."""
        self.observations.append(observation)

    def _parse_response(self, response_text: str) -> Tuple[str, Optional[str]]:
        """Parse LLM response to extract decision and reasoning."""
        # Try JSON parsing first
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                decision = data.get("decision", "")
                reasoning = data.get("reasoning", "")
                return str(decision), reasoning
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback: look for patterns
        decision_match = re.search(
            r'(?:decision|vote|choice|choose|pick|select)[:\s]*["\']?([^"\'\n,]+)',
            response_text,
            re.IGNORECASE,
        )
        if decision_match:
            return decision_match.group(1).strip(), response_text
        
        # Last resort: return the whole thing as the decision
        return response_text.strip()[:100], None

    def _generate_response(
        self,
        action: str,
        context: str,
        options: List[str],
        game_state: Dict[str, Any],
    ) -> Tuple[str, Optional[str]]:
        """Generate a response using the LLM.

        Args:
            action: The type of action requested
            context: Instructions/context from the green agent for this action
            options: Valid choices for this action
            game_state: Current game state from the green agent

        Returns:
            Tuple of (decision, reasoning)
        """
        if not self.system_prompt:
            raise ValueError("Role not assigned. Call assign_role first.")

        # Build action prompt using context from green agent
        action_prompt = get_action_prompt(
            action=action,
            context=context,
            options=options,
            game_state=game_state,
            observations=self.observations,
        )
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": action_prompt},
        ]
        
        # Add conversation history context
        if self.game_history:
            history_summary = "\n".join([
                f"- {h.get('action', '?')}: {h.get('decision', '?')}"
                for h in self.game_history[-5:]  # Last 5 actions
            ])
            messages.insert(1, {
                "role": "assistant",
                "content": f"My recent actions:\n{history_summary}",
            })
        
        # Dummy/Random mode for testing without API key
        if self.model == "dummy":
            import random
            decision = random.choice(options) if options else "pass"
            return decision, "Random choice in dummy mode"

        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Ensure client is initialized (will raise if no key and not dummy)
                _ = self.client
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature + (attempt * 0.1),
                    max_tokens=500,
                    response_format={"type": "json_object"} if "gpt-4" in self.model else None,
                )
                
                response_text = response.choices[0].message.content or ""
                decision, reasoning = self._parse_response(response_text)
                
                # Validate decision against options if provided
                if options and decision:
                    # Find best match
                    decision_lower = decision.lower().strip()
                    for opt in options:
                        if opt.lower().strip() in decision_lower or decision_lower in opt.lower():
                            return opt, reasoning
                    
                    # No match found, try another response
                    if attempt < self.max_retries - 1:
                        continue
                    
                    # Last attempt - return first option as fallback
                    logger.warning(f"Could not match decision '{decision}' to options, using first option")
                    return options[0], f"Fallback: {reasoning}"
                
                return decision, reasoning
                
            except Exception as e:
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
        
        # All retries failed
        if options:
            return options[0], f"Error fallback: {last_error}"
        return "", f"Error: {last_error}"

    def process_action(
        self,
        action: str,
        game_state: Dict[str, Any],
        options: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process an action request and return response.

        Args:
            action: The type of action requested (e.g., "vote", "debate")
            game_state: Current game state from the green agent
            options: Valid choices for this action
            context: Instructions from the green agent for this action

        Returns:
            Dict with "decision" and "reasoning" keys
        """
        options = options or []
        context = context or "Make your decision."

        # Add any announcements to observations
        for announcement in game_state.get("announcements", []):
            if announcement not in self.observations:
                self.observations.append(announcement)

        # Add player's own observations from game state
        for obs in game_state.get("your_observations", []):
            if obs not in self.observations:
                self.observations.append(obs)

        # Generate decision using context from green agent
        decision, reasoning = self._generate_response(
            action=action,
            context=context,
            options=options,
            game_state=game_state,
        )
        
        # Record in history
        self.game_history.append({
            "action": action,
            "round": game_state.get("round", 0),
            "decision": decision,
            "reasoning": reasoning,
        })
        
        logger.info(f"[{self.player_name}] {action}: {decision}")
        
        return {
            "decision": decision,
            "reasoning": reasoning,
        }


# Shared player instance for the server
_player_instance: Optional[LLMPlayer] = None


def get_player(model: str = "gpt-4o-mini") -> LLMPlayer:
    """Get or create the shared player instance."""
    global _player_instance
    if _player_instance is None:
        _player_instance = LLMPlayer(model=model)
    return _player_instance


def reset_player():
    """Reset the shared player instance."""
    global _player_instance
    if _player_instance:
        _player_instance.reset()
