# Copyright 2024 Google LLC
# Modifications for AgentBeats integration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Purple Agent A2A Server - Werewolf player agent for AgentBeats."""

import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

# Load env vars
from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from purple_agent.player import LLMPlayer, get_player, reset_player

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============ App State ============

class AppState:
    """Application state for the Purple Agent server."""

    def __init__(self):
        self.host: str = "0.0.0.0"
        self.port: int = 9010  # AgentBeats standard port for purple agent
        self.card_url: Optional[str] = None
        self.player: Optional[LLMPlayer] = None
        self.model: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        
        # Auto-fallback to dummy if no API key present
        if self.model != "dummy" and not os.environ.get("OPENAI_API_KEY"):
            # Check if using other providers that might use different keys, but generally:
            logger.warning("OPENAI_API_KEY not found. Defaulting to 'dummy' model for simulation.")
            self.model = "dummy"
        self.current_task_id: Optional[str] = None

    @property
    def base_url(self) -> str:
        if self.card_url:
            return self.card_url
        return f"http://{self.host}:{self.port}"


app_state = AppState()


# ============ Lifespan ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(f"🎭 Werewolf Purple Agent starting at {app_state.base_url}")
    logger.info(f"📋 Agent card available at {app_state.base_url}/info")
    logger.info(f"🤖 Using LLM model: {app_state.model}")
    app_state.player = get_player(model=app_state.model)
    yield
    logger.info("🎭 Werewolf Purple Agent shutting down")


# ============ FastAPI App ============

app = FastAPI(
    title="Werewolf Arena - Purple Agent",
    description="AgentBeats-compatible player agent for the Werewolf social deduction game",
    version="1.0.0",
    lifespan=lifespan,
)


# ============ Agent Card Endpoint ============

def _build_agent_card() -> Dict[str, Any]:
    """Build the agent card for this Purple Agent."""
    return {
        "name": "Werewolf Player Agent",
        "description": (
            "An LLM-powered player agent capable of playing any role "
            "in the Werewolf social deduction game."
        ),
        "version": "1.0.0",
        "url": app_state.base_url,
        "skills": [
            {
                "id": "werewolf-player",
                "name": "Werewolf Game Player",
                "description": "Plays roles in the Werewolf social deduction game (werewolf, villager, seer, doctor)",
                "tags": ["game", "player", "social-deduction", "werewolf"],
            }
        ],
        "capabilities": {
            "streaming": False,
            "roles": ["player"],
            "protocols": ["a2a"],
            "supported_game_roles": ["werewolf", "villager", "seer", "doctor"],
        },
        "default_input_modes": ["text"],
        "default_output_modes": ["text"],
        "model": app_state.model,
        "status": "ready" if app_state.player else "initializing",
    }


@app.get("/.well-known/agent-card.json")
async def get_agent_card() -> Dict[str, Any]:
    """A2A standard endpoint for agent card discovery."""
    return _build_agent_card()


@app.get("/info")
async def get_agent_info() -> Dict[str, Any]:
    """Return the agent card describing this Purple Agent's capabilities."""
    return _build_agent_card()


# ============ A2A Endpoint ============

class A2ARequest(BaseModel):
    """Generic A2A request envelope."""
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any] = {}
    id: Optional[str] = None


class A2AResponse(BaseModel):
    """Generic A2A response envelope."""
    jsonrpc: str = "2.0"
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[str] = None


@app.post("/a2a")
@app.post("/")  # Also handle at root for A2A client compatibility
async def handle_a2a(request: A2ARequest) -> Dict[str, Any]:
    """Main A2A protocol endpoint."""
    logger.info(f"Received A2A request: method={request.method}, id={request.id}")
    
    try:
        if request.method == "role_assignment":
            return await handle_role_assignment(request.params, request.id)
        elif request.method == "action_request":
            return await handle_action_request(request.params, request.id)
        elif request.method == "reset":
            return await handle_reset(request.id)
        else:
            return A2AResponse(
                id=request.id,
                error={"code": -32601, "message": f"Unknown method: {request.method}"},
            ).model_dump()
    except Exception as e:
        logger.exception(f"Error handling A2A request: {e}")
        return A2AResponse(
            id=request.id,
            error={"code": -32000, "message": str(e)},
        ).model_dump()


async def handle_role_assignment(params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
    """Handle role assignment from Green Agent."""
    if not app_state.player:
        return A2AResponse(
            id=request_id,
            error={"code": -32000, "message": "Player not initialized"},
        ).model_dump()
    
    # Extract parameters
    task_id = params.get("task_id")
    player_name = params.get("player_name")
    role = params.get("role")
    role_description = params.get("role_description", "")
    game_rules = params.get("game_rules", "")
    teammates = params.get("teammates")
    
    if not all([player_name, role]):
        return A2AResponse(
            id=request_id,
            error={"code": -32602, "message": "player_name and role required"},
        ).model_dump()
    
    # Reset if new task
    if app_state.current_task_id != task_id:
        reset_player()
        app_state.player = get_player(model=app_state.model)
        app_state.current_task_id = task_id
    
    # Assign role
    app_state.player.assign_role(
        player_name=player_name,
        role=role,
        role_description=role_description,
        game_rules=game_rules,
        teammates=teammates,
    )
    
    logger.info(f"Role assigned: {player_name} is {role}")
    
    return A2AResponse(
        id=request_id,
        result={
            "status": "ok",
            "message": f"Role {role} assigned to {player_name}",
        },
    ).model_dump()


async def handle_action_request(params: Dict[str, Any], request_id: Optional[str]) -> Dict[str, Any]:
    """Handle action request from Green Agent."""
    if not app_state.player:
        return A2AResponse(
            id=request_id,
            error={"code": -32000, "message": "Player not initialized"},
        ).model_dump()
    
    if not app_state.player.role:
        return A2AResponse(
            id=request_id,
            error={"code": -32000, "message": "Role not assigned yet"},
        ).model_dump()
    
    # Extract parameters
    action = params.get("action")
    game_state = params.get("game_state", {})
    options = params.get("options", [])
    context = params.get("context")
    
    if not action:
        return A2AResponse(
            id=request_id,
            error={"code": -32602, "message": "action required"},
        ).model_dump()
    
    # Process the action
    response = app_state.player.process_action(
        action=action,
        game_state=game_state,
        options=options,
        context=context,
    )
    
    return A2AResponse(
        id=request_id,
        result=response,
    ).model_dump()


async def handle_reset(request_id: Optional[str]) -> Dict[str, Any]:
    """Handle reset request - prepare for new game."""
    reset_player()
    app_state.player = get_player(model=app_state.model)
    app_state.current_task_id = None
    
    logger.info("Player reset for new game")
    
    return A2AResponse(
        id=request_id,
        result={"status": "ok", "message": "Player reset"},
    ).model_dump()


# ============ Additional REST Endpoints ============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "role": app_state.player.role if app_state.player else None,
        "player_name": app_state.player.player_name if app_state.player else None,
        "current_task": app_state.current_task_id,
    }


@app.post("/reset")
async def reset_endpoint():
    """REST endpoint to reset player state."""
    reset_player()
    app_state.player = get_player(model=app_state.model)
    app_state.current_task_id = None
    return {"status": "ok", "message": "Player reset"}


@app.get("/state")
async def get_state():
    """Get current player state (for debugging)."""
    if not app_state.player:
        return {"status": "not_initialized"}
    
    return {
        "player_name": app_state.player.player_name,
        "role": app_state.player.role,
        "team": app_state.player.team,
        "teammates": app_state.player.teammates,
        "observations_count": len(app_state.player.observations),
        "history_count": len(app_state.player.game_history),
        "current_task": app_state.current_task_id,
    }


# ============ Main Entry Point ============

def main():
    """Main entry point for the Purple Agent server."""
    parser = argparse.ArgumentParser(description="Werewolf Arena Purple Agent")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9010, help="Port to listen on (AgentBeats standard: 9010)")
    parser.add_argument("--card-url", help="URL to advertise in agent card")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    app_state.host = args.host
    app_state.port = args.port
    app_state.card_url = args.card_url
    app_state.model = args.model
    
    uvicorn.run(
        "purple_agent.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
