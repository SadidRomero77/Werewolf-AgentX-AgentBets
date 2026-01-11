# Copyright 2024 Google LLC
# Modifications for AgentBeats integration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""A2A HTTP Client for communicating with Purple Agents."""

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

from green_agent.models import (
    A2AMessage,
    A2AResponse,
    ActionRequest,
    ActionResponse,
    ActionType,
    RoleAssignment,
    GameState,
)

logger = logging.getLogger(__name__)


class A2AClientError(Exception):
    """Exception raised when A2A communication fails."""
    pass


class A2AClient:
    """HTTP client for A2A protocol communication with Purple Agents."""

    def __init__(
        self,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: Optional[httpx.AsyncClient] = None

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        """Normalize endpoint URL by removing trailing slashes."""
        return endpoint.rstrip("/")

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def _send_message(
        self,
        endpoint: str,
        method: str,
        params: Dict[str, Any],
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send an A2A message and get response."""
        if not self._client:
            raise A2AClientError("Client not initialized. Use async context manager.")

        # Normalize endpoint to avoid double slashes
        endpoint = self._normalize_endpoint(endpoint)

        message = A2AMessage(
            method=method,
            params=params,
            id=message_id,
        )

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self._client.post(
                    f"{endpoint}/a2a",
                    json=message.model_dump(),
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                
                result = response.json()
                if "error" in result and result["error"]:
                    raise A2AClientError(f"A2A error: {result['error']}")
                
                return result.get("result", {})
                
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"Timeout on attempt {attempt + 1}/{self.max_retries} "
                    f"for {endpoint}: {e}"
                )
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(
                    f"HTTP error on attempt {attempt + 1}/{self.max_retries} "
                    f"for {endpoint}: {e}"
                )
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error communicating with {endpoint}: {e}")
                break

            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))

        raise A2AClientError(
            f"Failed to communicate with {endpoint} after {self.max_retries} attempts: {last_error}"
        )

    async def get_agent_info(self, endpoint: str) -> Dict[str, Any]:
        """Get agent card/info from a Purple Agent."""
        if not self._client:
            raise A2AClientError("Client not initialized. Use async context manager.")

        # Normalize endpoint to avoid double slashes
        endpoint = self._normalize_endpoint(endpoint)

        try:
            response = await self._client.get(f"{endpoint}/info")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise A2AClientError(f"Failed to get agent info from {endpoint}: {e}")

    async def assign_role(
        self,
        endpoint: str,
        assignment: RoleAssignment,
    ) -> Dict[str, Any]:
        """Send role assignment to a Purple Agent."""
        return await self._send_message(
            endpoint=endpoint,
            method="role_assignment",
            params=assignment.model_dump(),
            message_id=f"{assignment.task_id}_role_{assignment.player_name}",
        )

    async def request_action(
        self,
        endpoint: str,
        request: ActionRequest,
    ) -> ActionResponse:
        """Request an action from a Purple Agent."""
        result = await self._send_message(
            endpoint=endpoint,
            method="action_request",
            params=request.model_dump(),
            message_id=f"{request.task_id}_{request.action}_{request.player_name}",
        )

        return ActionResponse(
            task_id=request.task_id,
            player_name=request.player_name,
            action=request.action,
            decision=result.get("decision", ""),
            reasoning=result.get("reasoning"),
        )

    async def request_actions_parallel(
        self,
        endpoints: Dict[str, str],  # player_name -> endpoint
        requests: Dict[str, ActionRequest],  # player_name -> request
    ) -> Dict[str, ActionResponse]:
        """Request actions from multiple Purple Agents in parallel."""
        tasks = {}
        for player_name, request in requests.items():
            endpoint = endpoints.get(player_name)
            if endpoint:
                tasks[player_name] = self.request_action(endpoint, request)

        results = await asyncio.gather(
            *[tasks[name] for name in tasks],
            return_exceptions=True,
        )

        responses = {}
        for player_name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to get action from {player_name}: {result}")
                # Return a default/error response
                responses[player_name] = ActionResponse(
                    task_id=requests[player_name].task_id,
                    player_name=player_name,
                    action=requests[player_name].action,
                    decision="",  # Empty decision indicates failure
                    reasoning=f"Error: {str(result)}",
                )
            else:
                responses[player_name] = result

        return responses


async def verify_agent_connectivity(
    endpoints: Dict[str, str],
    timeout: float = 10.0,
) -> Dict[str, bool]:
    """Verify that all Purple Agents are reachable."""
    async with A2AClient(timeout=timeout, max_retries=1) as client:
        results = {}
        for player_name, endpoint in endpoints.items():
            try:
                await client.get_agent_info(endpoint)
                results[player_name] = True
                logger.info(f"✓ Agent {player_name} at {endpoint} is reachable")
            except Exception as e:
                results[player_name] = False
                logger.warning(f"✗ Agent {player_name} at {endpoint} unreachable: {e}")
        return results
