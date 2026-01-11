# Copyright 2024 Google LLC
# Modifications for AgentBeats integration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Green Agent A2A Server - FastAPI-based evaluator for Werewolf Arena."""

import argparse
import asyncio
import logging
import os
import uuid
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, set_key, find_dotenv

from green_agent.models import (
    A2AMessage,
    A2AResponse,
    AgentCard,
    AgentCapabilities,
    AssessmentConfig,
    AssessmentRequest,
    AssessmentResult,
    AssessmentSpec,
    TaskUpdate,
)
from green_agent.orchestrator import GameOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def is_running_in_docker() -> bool:
    """Detect if we're running inside a Docker container."""
    # Check for .dockerenv file or cgroup
    return os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')


def get_agent_url(player_index: int) -> str:
    """Get the correct URL for a purple agent based on environment."""
    # Use AgentBeats standard port 9010 as base for purple agents
    port = 9010 + player_index
    if is_running_in_docker():
        # Use Docker service names
        return f"http://purple-agent-{player_index + 1}:{port}/a2a"
    else:
        # Use localhost for local development
        return f"http://localhost:{port}/a2a"


# ============ WebSocket Manager ============
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


# ============ App State ============

class AppState:
    """Application state for the Green Agent server."""

    def __init__(self):
        # Read from environment variables (set by main() before uvicorn reimports)
        self.host: str = os.environ.get("AGENT_HOST", "0.0.0.0")
        self.port: int = int(os.environ.get("AGENT_PORT", "9009"))
        self.card_url: Optional[str] = os.environ.get("AGENT_CARD_URL")
        self.active_assessments: Dict[str, GameOrchestrator] = {}
        self.completed_results: Dict[str, AssessmentResult] = {}
        self.task_updates: Dict[str, List[TaskUpdate]] = {}
        self.ws_manager = ConnectionManager()

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
    logger.info(f"🐺 Werewolf Arena Green Agent starting at {app_state.base_url}")
    logger.info(f"📋 Agent card available at {app_state.base_url}/info")
    yield
    logger.info("🐺 Werewolf Arena Green Agent shutting down")


# ============ FastAPI App ============

app = FastAPI(
    title="Werewolf Arena - Green Agent",
    description="AgentBeats-compatible evaluator for the Werewolf social deduction game",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Agent Card Endpoint ============

def _build_agent_card() -> Dict[str, Any]:
    """Build the agent card for this Green Agent."""
    return {
        "name": "Werewolf Arena Evaluator",
        "description": (
            "A multi-agent social deduction game evaluator. "
            "Tests LLM agents' ability to reason socially, deceive, detect deception, "
            "and collaborate in the classic Werewolf game."
        ),
        "version": "1.0.0",
        "url": app_state.base_url,
        "skills": [
            {
                "id": "werewolf-game",
                "name": "Werewolf Game Evaluation",
                "description": "Evaluates AI agents playing the Werewolf social deduction game",
                "tags": ["game", "evaluation", "social-deduction", "multi-agent"],
            }
        ],
        "capabilities": {
            "streaming": False,
            "roles": ["evaluator"],
            "protocols": ["a2a"],
        },
        "default_input_modes": ["text"],
        "default_output_modes": ["text", "data"],
        "assessment": {
            "min_participants": 5,
            "max_participants": 8,
            "supported_roles": ["werewolf", "villager", "seer", "doctor"],
        },
    }


@app.get("/.well-known/agent-card.json")
async def get_agent_card() -> Dict[str, Any]:
    """A2A standard endpoint for agent card discovery."""
    return _build_agent_card()


@app.get("/info")
async def get_agent_info() -> Dict[str, Any]:
    """Return the agent card describing this Green Agent's capabilities."""
    return _build_agent_card()


# ============ A2A Endpoint ============

class A2ARequest(BaseModel):
    """Generic A2A request envelope."""
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any] = {}
    id: Optional[str] = None


@app.post("/a2a")
@app.post("/")  # Also handle at root for A2A client compatibility
async def handle_a2a(request: A2ARequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Main A2A protocol endpoint."""
    logger.info(f"Received A2A request: method={request.method}, id={request.id}")

    try:
        # A2A standard message/send method
        if request.method == "message/send":
            return await handle_message_send(request.params, request.id, background_tasks)
        # Legacy methods for backwards compatibility
        elif request.method == "assessment_request":
            return await handle_assessment_request(request.params, background_tasks)
        elif request.method == "get_status":
            return await handle_get_status(request.params)
        elif request.method == "get_result":
            return await handle_get_result(request.params)
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


async def handle_message_send(
    params: Dict[str, Any],
    request_id: Optional[str],
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Handle A2A standard message/send method."""
    logger.info(f"message/send params: {json.dumps(params, indent=2)}")

    message = params.get("message", {})

    # Extract task info from message
    task_id = message.get("taskId") or message.get("task_id") or str(uuid.uuid4())
    context_id = message.get("contextId") or task_id

    # Get participants from message parts or context
    parts = message.get("parts", [])

    # Try to extract participants from the message
    participants = {}
    config_dict = {}

    for part in parts:
        if part.get("kind") == "data":
            data = part.get("data", {})
            if "participants" in data:
                participants = data["participants"]
            if "config" in data:
                config_dict = data["config"]
        elif part.get("kind") == "text":
            # Text part might contain JSON with participants
            text = part.get("text", "")
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    if "participants" in data:
                        participants = data["participants"]
                    if "config" in data:
                        config_dict = data["config"]
            except json.JSONDecodeError:
                pass

    # Check if participants are passed at the top level of params
    if not participants and "participants" in params:
        participants = params["participants"]

    logger.info(f"Extracted participants: {participants}")

    # If no participants, return error in A2A format
    if not participants or len(participants) < 5:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "kind": "task",
                "id": task_id,
                "contextId": context_id,
                "status": {"state": "failed"},
                "artifacts": [{
                    "artifactId": f"{task_id}_error",
                    "parts": [{
                        "kind": "text",
                        "text": f"Error: Minimum 5 participants required, got {len(participants)}. Participants: {list(participants.keys()) if participants else 'none'}"
                    }]
                }]
            }
        }

    # Initialize task tracking
    app_state.task_updates[task_id] = []

    def on_update(update: TaskUpdate):
        app_state.task_updates[task_id].append(update)

    # Create config
    config = AssessmentConfig(**config_dict) if config_dict else AssessmentConfig()
    config.num_players = len(participants)

    # Create orchestrator
    orchestrator = GameOrchestrator(
        task_id=task_id,
        participants=participants,
        config=config,
        on_task_update=on_update,
        event_callback=app_state.ws_manager.broadcast,
    )
    app_state.active_assessments[task_id] = orchestrator

    # Check if blocking mode is requested
    configuration = params.get("configuration", {})
    is_blocking = configuration.get("blocking", False)

    if is_blocking:
        # Run game synchronously for blocking requests
        logger.info(f"Running assessment {task_id} synchronously (blocking mode)")
        try:
            result = await orchestrator.run_game()
            app_state.completed_results[task_id] = result
            logger.info(f"Assessment {task_id} completed. Winner: {result.winner}")

            # Return completed task with results
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "kind": "task",
                    "id": task_id,
                    "contextId": context_id,
                    "status": {"state": "completed"},
                    "artifacts": [{
                        "artifactId": f"{task_id}_results",
                        "parts": [
                            {
                                "kind": "text",
                                "text": f"Game completed! Winner: {result.winner}"
                            },
                            {
                                "kind": "data",
                                "data": {
                                    "winner": result.winner,
                                    "rounds_played": result.rounds_played,
                                    "game_log": result.game_log[-10:] if result.game_log else [],
                                    "scores": [score.model_dump() for score in result.scores] if result.scores else [],
                                }
                            }
                        ]
                    }]
                }
            }
        except Exception as e:
            logger.exception(f"Assessment {task_id} failed: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "kind": "task",
                    "id": task_id,
                    "contextId": context_id,
                    "status": {"state": "failed"},
                    "artifacts": [{
                        "artifactId": f"{task_id}_error",
                        "parts": [{
                            "kind": "text",
                            "text": f"Game failed: {str(e)}"
                        }]
                    }]
                }
            }
    else:
        # Run game in background for non-blocking requests
        background_tasks.add_task(run_assessment, task_id, orchestrator)
        logger.info(f"Started assessment {task_id} with {len(participants)} participants via message/send")

        # Return A2A standard response format (Task object)
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "kind": "task",
                "id": task_id,
                "contextId": context_id,
                "status": {"state": "working"},
                "artifacts": [{
                    "artifactId": f"{task_id}_start",
                    "parts": [{
                        "kind": "text",
                        "text": f"Game started with {len(participants)} players: {list(participants.keys())}"
                    }]
                }]
            }
        }


async def handle_assessment_request(
    params: Dict[str, Any],
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Handle a new assessment request."""
    # Parse request
    task_id = params.get("task_id", str(uuid.uuid4()))
    participants = params.get("participants", {})
    config_dict = params.get("config", {})
    
    # Validate participants
    num_participants = len(participants)
    if num_participants < 5:
        return A2AResponse(
            error={
                "code": -32602,
                "message": f"Minimum 5 participants required, got {num_participants}",
            },
        ).model_dump()
    
    if num_participants > 8:
        return A2AResponse(
            error={
                "code": -32602,
                "message": f"Maximum 8 participants allowed, got {num_participants}",
            },
        ).model_dump()
    
    # Create config
    config = AssessmentConfig(**config_dict) if config_dict else AssessmentConfig()
    config.num_players = num_participants
    
    # Initialize task tracking
    app_state.task_updates[task_id] = []
    
    def on_update(update: TaskUpdate):
        app_state.task_updates[task_id].append(update)
    
    # Create orchestrator
    orchestrator = GameOrchestrator(
        task_id=task_id,
        participants=participants,
        config=config,
        on_task_update=on_update,
        event_callback=app_state.ws_manager.broadcast,
    )
    app_state.active_assessments[task_id] = orchestrator
    
    # Run game in background
    background_tasks.add_task(run_assessment, task_id, orchestrator)
    
    logger.info(f"Started assessment {task_id} with {num_participants} participants")
    
    return A2AResponse(
        result={
            "task_id": task_id,
            "status": "started",
            "message": f"Assessment started with {num_participants} players",
        },
    ).model_dump()


# duplicate import of dotenv to ensure it is loaded
from dotenv import load_dotenv
load_dotenv()

async def run_assessment(task_id: str, orchestrator: GameOrchestrator):
    """Run an assessment in the background."""
    try:
        result = await orchestrator.run_game()
        app_state.completed_results[task_id] = result
        logger.info(f"Assessment {task_id} completed. Winner: {result.winner}")
    except Exception as e:
        logger.exception(f"Assessment {task_id} failed: {e}")
        
        # Broadcast error to frontend
        await app_state.ws_manager.broadcast({
            "type": "system_error",
            "timestamp": datetime.now().isoformat(),
            "payload": {
                "error": str(e),
                "task_id": task_id
            }
        })
        
        # Store error result
        app_state.completed_results[task_id] = AssessmentResult(
            task_id=task_id,
            winner="error",
            rounds_played=0,
            aggregate_metrics={"error": -1.0}, # Use -1.0 to indicate error in metrics
        )
    finally:
        # Clean up active assessment
        if task_id in app_state.active_assessments:
            del app_state.active_assessments[task_id]


async def handle_get_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get status of an assessment."""
    task_id = params.get("task_id")
    if not task_id:
        return A2AResponse(
            error={"code": -32602, "message": "task_id required"},
        ).model_dump()
    
    # Check if completed
    if task_id in app_state.completed_results:
        return A2AResponse(
            result={
                "task_id": task_id,
                "status": "completed",
                "winner": app_state.completed_results[task_id].winner,
            },
        ).model_dump()
    
    # Check if active
    if task_id in app_state.active_assessments:
        orchestrator = app_state.active_assessments[task_id]
        updates = app_state.task_updates.get(task_id, [])
        latest_update = updates[-1].message if updates else "Starting..."
        
        return A2AResponse(
            result={
                "task_id": task_id,
                "status": "running",
                "round": orchestrator.current_round,
                "phase": orchestrator.current_phase.value,
                "latest_update": latest_update,
            },
        ).model_dump()
    
    return A2AResponse(
        error={"code": -32602, "message": f"Unknown task: {task_id}"},
    ).model_dump()


async def handle_get_result(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get result of a completed assessment."""
    task_id = params.get("task_id")
    if not task_id:
        return A2AResponse(
            error={"code": -32602, "message": "task_id required"},
        ).model_dump()
    
    if task_id not in app_state.completed_results:
        return A2AResponse(
            error={"code": -32602, "message": f"No result for task: {task_id}"},
        ).model_dump()
    
    result = app_state.completed_results[task_id]
    return A2AResponse(
        result=result.model_dump(),
    ).model_dump()


# ============ Additional REST Endpoints ============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_assessments": len(app_state.active_assessments),
        "completed_assessments": len(app_state.completed_results),
    }


@app.get("/assessments")
async def list_assessments():
    """List all assessments."""
    return {
        "active": list(app_state.active_assessments.keys()),
        "completed": list(app_state.completed_results.keys()),
    }


@app.get("/assessments/{task_id}")
async def get_assessment(task_id: str):
    """Get details of a specific assessment."""
    if task_id in app_state.completed_results:
        return app_state.completed_results[task_id].model_dump()
    
    if task_id in app_state.active_assessments:
        orch = app_state.active_assessments[task_id]
        return {
            "task_id": task_id,
            "status": "running",
            "round": orch.current_round,
            "phase": orch.current_phase.value,
            "alive_players": orch.alive_players,
            "updates": [u.model_dump() for u in app_state.task_updates.get(task_id, [])],
        }
    
    raise HTTPException(status_code=404, detail=f"Assessment not found: {task_id}")


@app.get("/assessments/{task_id}/updates")
async def get_assessment_updates(task_id: str):
    """Get task updates for an assessment."""
    updates = app_state.task_updates.get(task_id, [])
    return {
        "task_id": task_id,
        "updates": [u.model_dump() for u in updates],
    }


# ============ Frontend Integration Endpoints ============

class ConfigUpdate(BaseModel):
    OPENAI_API_BASE: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: Optional[str] = None

@app.post("/config")
async def update_config(config: ConfigUpdate):
    """Update LLM configuration in .env file."""
    env_file = find_dotenv()
    if not env_file:
        return JSONResponse({"error": ".env file not found"}, status_code=404)

    try:
        if config.OPENAI_API_BASE:
            set_key(env_file, "OPENAI_API_BASE", config.OPENAI_API_BASE)
            os.environ["OPENAI_API_BASE"] = config.OPENAI_API_BASE
        if config.OPENAI_API_KEY:
            set_key(env_file, "OPENAI_API_KEY", config.OPENAI_API_KEY)
            os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
        if config.LLM_MODEL:
            set_key(env_file, "LLM_MODEL", config.LLM_MODEL)
            os.environ["LLM_MODEL"] = config.LLM_MODEL
            
        return {"status": "success", "message": "Configuration updated"}
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

class StartGameRequest(BaseModel):
    num_players: int = 5

@app.post("/start_game")
async def start_game_manual(request: StartGameRequest, background_tasks: BackgroundTasks):
    """Manually start a game from the frontend with specified player count."""
    request_id = f"manual_{uuid.uuid4().hex[:8]}"
    
    num_players = max(5, min(8, request.num_players))
    
    # Generate participants based on count (1-based indexing)
    participants = {
        f"Player_{i+1}": get_agent_url(i) for i in range(num_players)
    }
    
    params = {
        "task_id": request_id,
        "participants": participants,
        "config": {
            "num_players": num_players
        }
    }
    
    logger.info(f"Starting manual game with {num_players} players")
    return await handle_assessment_request(params, background_tasks)
    
    # We must manually attach the tasks to the response or run them
    # Since we are returning a JSON response, we can't easily attach BackgroundTasks object
    # unless we change the signature of this endpoint to accept BackgroundTasks
    # and pass it through.
    
    # But wait, handle_assessment_request returns a dict (A2AResponse dump).
    # It adds tasks to the passed `background_tasks`.
    # So if we accept `background_tasks` in THIS endpoint, we can pass it down.
    
    return response

@app.post("/start_game_simple")
async def start_game_simple(background_tasks: BackgroundTasks):
    """Simpler endpoint that properly injects background tasks."""
    # This is the one we will use
    request_id = f"manual_{uuid.uuid4().hex[:8]}"
    
    # Defaults: 5 players
    participants = {
        f"Player_{i+1}": get_agent_url(i) for i in range(5)
    }
    
    params = {
        "task_id": request_id,
        "participants": participants,
        "config": {"num_players": 5}
    }
    
    return await handle_assessment_request(params, background_tasks)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await app_state.ws_manager.connect(websocket)
    try:
        while True:
            # Keep alive / listen for client messages
            # We don't really expect much input from client yet, maybe "ping"
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        app_state.ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# ============ Main Entry Point ============

def main():
    """Main entry point for the Green Agent server."""
    parser = argparse.ArgumentParser(description="Werewolf Arena Green Agent")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9009, help="Port to listen on (AgentBeats standard: 9009)")
    parser.add_argument("--card-url", help="URL to advertise in agent card")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    # Set environment variables so they're available when uvicorn reimports the module
    if args.card_url:
        os.environ["AGENT_CARD_URL"] = args.card_url
    os.environ["AGENT_HOST"] = args.host
    os.environ["AGENT_PORT"] = str(args.port)

    uvicorn.run(
        "green_agent.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
