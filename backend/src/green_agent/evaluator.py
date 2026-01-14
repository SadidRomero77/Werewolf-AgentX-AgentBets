# Copyright 2024 Google LLC
# Modifications for AgentBeats integration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""
Qualitative Evaluator for Werewolf Arena

Uses LLM-as-a-Judge with G-Eval methodology to analyze agent interactions 
and explain WHY an agent performed better.

Academic References:
- Zheng et al. 2023 "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS)
- Liu et al. 2023 "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"
- Light et al. 2023 "AvalonBench: Evaluating LLMs Playing the Game of Avalon"
- Li et al. 2024 "LLMs-as-Judges: A Comprehensive Survey"
"""

import json
import logging
import os
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# ============================================================================
# RUBRICS - Explicit scoring criteria based on academic standards
# Reference: G-Eval (Liu et al. 2023), AvalonBench (Light et al. 2023)
# ============================================================================

RUBRICS = {
    "reasoning_quality": {
        "name": "Reasoning Quality",
        "definition": "Quality of logical deduction, inference, and analysis during gameplay",
        "weight": 0.25,
        "scale": {
            1: {"label": "Very Poor", "description": "No reasoning, random decisions"},
            2: {"label": "Poor", "description": "Contradictory reasoning, ignores available information"},
            3: {"label": "Below Average", "description": "Superficial reasoning, follows crowd blindly"},
            4: {"label": "Slightly Below Average", "description": "Basic reasoning with notable gaps"},
            5: {"label": "Average", "description": "Adequate reasoning, uses obvious information"},
            6: {"label": "Slightly Above Average", "description": "Good reasoning, makes valid deductions"},
            7: {"label": "Good", "description": "Solid reasoning, anticipates some moves"},
            8: {"label": "Very Good", "description": "Strong deductions, identifies patterns"},
            9: {"label": "Excellent", "description": "Exceptional analysis, multi-step planning"},
            10: {"label": "Outstanding", "description": "Near-optimal reasoning, predicts opponent moves"},
        },
    },
    "persuasion_effectiveness": {
        "name": "Persuasion Effectiveness",
        "definition": "Ability to influence other players' decisions through argumentation",
        "weight": 0.20,
        "scale": {
            1: {"label": "Very Poor", "description": "Arguments ignored or counterproductive"},
            2: {"label": "Poor", "description": "Weak arguments, no influence on others"},
            3: {"label": "Below Average", "description": "Occasional influence, mostly ignored"},
            4: {"label": "Slightly Below Average", "description": "Some persuasive moments but inconsistent"},
            5: {"label": "Average", "description": "Moderate influence on group decisions"},
            6: {"label": "Slightly Above Average", "description": "Regularly contributes to consensus"},
            7: {"label": "Good", "description": "Often sways votes and opinions"},
            8: {"label": "Very Good", "description": "Strong persuader, builds coalitions"},
            9: {"label": "Excellent", "description": "Dominates discussions, shapes narratives"},
            10: {"label": "Outstanding", "description": "Master persuader, controls group dynamics"},
        },
    },
    "deception_skill": {
        "name": "Deception Skill",
        "definition": "Ability to hide werewolf identity and mislead villagers (Werewolf only)",
        "weight": 0.25,
        "scale": {
            1: {"label": "Exposed", "description": "Immediately detected, obvious werewolf behavior"},
            2: {"label": "Very Suspicious", "description": "Detected within first round"},
            3: {"label": "Suspicious", "description": "Generates strong suspicion early"},
            4: {"label": "Somewhat Suspicious", "description": "Notable inconsistencies in behavior"},
            5: {"label": "Neutral", "description": "Blends in but doesn't deflect suspicion"},
            6: {"label": "Somewhat Hidden", "description": "Avoids being primary suspect"},
            7: {"label": "Well Hidden", "description": "Deflects suspicion to others"},
            8: {"label": "Excellent Deceiver", "description": "Builds false trust, never suspected"},
            9: {"label": "Master Deceiver", "description": "Manipulates village against innocents"},
            10: {"label": "Perfect Deception", "description": "Completely trusted, leads village astray"},
        },
    },
    "detection_ability": {
        "name": "Detection Ability",
        "definition": "Ability to identify werewolves through behavioral analysis (Villager team only)",
        "weight": 0.25,
        "scale": {
            1: {"label": "Blind", "description": "Cannot identify werewolves, trusts them"},
            2: {"label": "Very Poor", "description": "Consistently votes against innocents"},
            3: {"label": "Poor", "description": "Rarely identifies werewolves correctly"},
            4: {"label": "Below Average", "description": "Occasional correct identification by luck"},
            5: {"label": "Average", "description": "Identifies some werewolves with effort"},
            6: {"label": "Above Average", "description": "Good behavioral analysis"},
            7: {"label": "Good Detective", "description": "Regularly identifies werewolves"},
            8: {"label": "Very Good", "description": "Early and accurate detection"},
            9: {"label": "Excellent", "description": "Deduces roles with minimal information"},
            10: {"label": "Sherlock", "description": "Near-perfect detection from round 1"},
        },
    },
    "adaptation": {
        "name": "Strategic Adaptation",
        "definition": "Ability to adjust strategy based on new information and game state",
        "weight": 0.15,
        "scale": {
            1: {"label": "Rigid", "description": "Never changes approach despite failures"},
            2: {"label": "Very Rigid", "description": "Rarely adjusts, repeats mistakes"},
            3: {"label": "Somewhat Rigid", "description": "Slow to adapt to new information"},
            4: {"label": "Slightly Inflexible", "description": "Makes some adjustments but delayed"},
            5: {"label": "Average", "description": "Adapts when situation is obvious"},
            6: {"label": "Moderately Adaptive", "description": "Adjusts strategy proactively"},
            7: {"label": "Good", "description": "Quick to incorporate new information"},
            8: {"label": "Very Good", "description": "Anticipates need for adaptation"},
            9: {"label": "Excellent", "description": "Fluid strategy, always optimal response"},
            10: {"label": "Outstanding", "description": "Predicts game flow, pre-adapts"},
        },
    },
    "consistency": {
        "name": "Logical Consistency",
        "definition": "Maintaining coherent narrative and avoiding self-contradiction",
        "weight": 0.15,
        "scale": {
            1: {"label": "Incoherent", "description": "Constant contradictions, no logic"},
            2: {"label": "Very Inconsistent", "description": "Major contradictions in statements"},
            3: {"label": "Inconsistent", "description": "Frequent contradictions"},
            4: {"label": "Somewhat Inconsistent", "description": "Notable but infrequent contradictions"},
            5: {"label": "Average", "description": "Mostly consistent with minor issues"},
            6: {"label": "Above Average", "description": "Consistent narrative maintained"},
            7: {"label": "Good", "description": "Strong consistency throughout"},
            8: {"label": "Very Good", "description": "Excellent narrative coherence"},
            9: {"label": "Excellent", "description": "Perfect consistency, credible story"},
            10: {"label": "Outstanding", "description": "Unassailable logical coherence"},
        },
    },
}


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class SkillScore:
    """Score for a specific skill with evidence and explanation."""
    skill_name: str
    score: int  # 1-10
    rubric_level: str
    evidence: List[str]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "score_normalized": round(self.score / 10.0, 2),
            "rubric_level": self.rubric_level,
            "evidence": self.evidence,
            "explanation": self.explanation,
        }


@dataclass
class AgentEvaluation:
    """Complete evaluation of a single agent's performance."""
    agent_name: str
    role: str
    team: str
    won: bool
    survived: bool
    
    # Skill scores
    skill_scores: Dict[str, SkillScore] = field(default_factory=dict)
    
    # Composite scores
    overall_score: float = 0.0  # 0-100
    rank: int = 0
    
    # Qualitative feedback
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    key_moments: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "role": self.role,
            "team": self.team,
            "won": self.won,
            "survived": self.survived,
            "overall_score": round(self.overall_score, 2),
            "rank": self.rank,
            "skill_scores": {
                name: score.to_dict() 
                for name, score in self.skill_scores.items()
            },
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "key_moments": self.key_moments,
            "improvement_suggestions": self.improvement_suggestions,
        }


@dataclass
class GameEvaluation:
    """Complete evaluation of a game with validation metadata."""
    game_id: str
    winner: str
    total_rounds: int
    
    # Best player determination
    best_player: str
    best_player_justification: str
    
    # Per-agent evaluations
    agent_evaluations: List[AgentEvaluation]
    
    # Game-level analysis
    key_turning_points: List[str]
    reasoning_quality_comparison: str
    model_insights: str
    
    # Validation metadata (for academic rigor)
    validation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format compatible with existing artifact structure."""
        return {
            "methodology": {
                "framework": "LLM-as-a-Judge with G-Eval",
                "rubric_version": "1.0",
                "judge_model": os.getenv("EVALUATOR_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini")),
                "temperature": 0.3,
                "references": [
                    "Zheng et al. 2023 - Judging LLM-as-a-Judge (NeurIPS)",
                    "Liu et al. 2023 - G-Eval",
                    "Light et al. 2023 - AvalonBench",
                    "Li et al. 2024 - LLMs-as-Judges Survey",
                ],
            },
            "best_player": {
                "name": self.best_player,
                "justification": self.best_player_justification,
            },
            "rankings": [
                {
                    "rank": e.rank,
                    "agent_name": e.agent_name,
                    "role": e.role,
                    "overall_score": round(e.overall_score, 2),
                }
                for e in sorted(self.agent_evaluations, key=lambda x: x.rank)
            ],
            "agent_evaluations": [e.to_dict() for e in self.agent_evaluations],
            "game_analysis": {
                "key_turning_points": self.key_turning_points,
                "reasoning_quality_comparison": self.reasoning_quality_comparison,
                "model_insights": self.model_insights,
            },
            "validation": self.validation if self.validation else {
                "validated": False,
                "note": "Single-run evaluation. Set validate=True for inter-rater reliability."
            },
        }


# ============================================================================
# EVALUATION PROMPTS WITH G-EVAL METHODOLOGY
# ============================================================================

def build_rubric_text(skill_name: str) -> str:
    """Build formatted rubric text for a skill."""
    rubric = RUBRICS.get(skill_name, {})
    if not rubric:
        return ""
    
    lines = [f"### {rubric['name']}", f"Definition: {rubric['definition']}", "", "Scoring Scale:"]
    for score, details in rubric.get("scale", {}).items():
        lines.append(f"  {score}: {details['label']} - {details['description']}")
    
    return "\n".join(lines)


AGENT_EVALUATION_PROMPT = """You are an expert judge evaluating AI agents in the social deduction game Werewolf.
Your evaluation must follow the G-Eval methodology with explicit rubrics.

## Game Context
- Winner: {winner}
- Total Rounds: {total_rounds}
- All Players and Roles: {player_roles}

## Agent Being Evaluated
- Name: {agent_name}
- Role: {role}
- Team: {team}
- Won: {won}
- Survived: {survived}

## Agent's Complete Action Log (decisions with reasoning)
{agent_actions}

## Agent's Debate Statements
{debate_statements}

## Voting Record
{voting_history}

---

## EVALUATION RUBRICS

{rubrics}

---

## EVALUATION INSTRUCTIONS (Chain-of-Thought)

For each applicable skill, follow these steps:
1. Review the agent's actions and statements carefully
2. Identify specific evidence (quotes, decisions, timestamps)
3. Compare behavior against the rubric scale
4. Assign a score (1-10) with clear justification

For {role} role, evaluate these skills:
- reasoning_quality (all roles)
- persuasion_effectiveness (all roles)
- {role_specific_skill} ({role} specific)
- adaptation (all roles)
- consistency (all roles)

## OUTPUT FORMAT (JSON only, no markdown)

{{
    "skill_scores": {{
        "reasoning_quality": {{
            "score": <1-10>,
            "rubric_level": "<label from rubric>",
            "evidence": ["<specific quote or action from the log>", "..."],
            "explanation": "<why this score based on rubric>"
        }},
        "persuasion_effectiveness": {{
            "score": <1-10>,
            "rubric_level": "<label>",
            "evidence": ["<specific evidence>"],
            "explanation": "<reasoning>"
        }},
        "{role_specific_skill}": {{
            "score": <1-10>,
            "rubric_level": "<label>",
            "evidence": ["<specific evidence>"],
            "explanation": "<reasoning>"
        }},
        "adaptation": {{
            "score": <1-10>,
            "rubric_level": "<label>",
            "evidence": ["<specific evidence>"],
            "explanation": "<reasoning>"
        }},
        "consistency": {{
            "score": <1-10>,
            "rubric_level": "<label>",
            "evidence": ["<specific evidence>"],
            "explanation": "<reasoning>"
        }}
    }},
    "overall_score": <0-100 weighted average>,
    "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"],
    "key_moments": ["<Round X: notable decision or mistake>"],
    "improvement_suggestions": ["<specific actionable suggestion>"]
}}
"""


COMPARATIVE_EVALUATION_PROMPT = """You are an expert judge determining the BEST PLAYER in a Werewolf game.

## Game Summary
- Winner: {winner}
- Total Rounds: {total_rounds}
- Players: {player_list}

## Individual Evaluation Summaries
{individual_summaries}

## Key Game Events
{game_events}

---

## YOUR TASK

Determine who played BEST, regardless of whether their team won.
A werewolf who played brilliantly but lost due to circumstances can still be "best player".

## EVALUATION CRITERIA

1. **Quality of Reasoning**: Sophisticated analysis vs simple heuristics
2. **Decision Impact**: How much their actions influenced the outcome
3. **Role Execution**: How well they fulfilled their role's objectives
4. **Adaptability**: Responding to changing game state
5. **Strategic Consistency**: Maintaining logical coherent play

## IMPORTANT NOTES

- Best player ≠ winning player necessarily
- Consider the difficulty of each role
- Evaluate based on QUALITY of play, not just outcome
- A player who made excellent decisions but lost to bad luck may be best

## OUTPUT FORMAT (JSON only, no markdown)

{{
    "best_player": "<agent_name>",
    "best_player_justification": "<3-5 detailed sentences explaining WHY this player was best, citing specific decisions and moments>",
    "player_ranking": [
        {{"rank": 1, "agent_name": "...", "score": <0-100>, "summary": "<one sentence>"}},
        {{"rank": 2, "agent_name": "...", "score": <0-100>, "summary": "<one sentence>"}},
        {{"rank": 3, "agent_name": "...", "score": <0-100>, "summary": "<one sentence>"}},
        {{"rank": 4, "agent_name": "...", "score": <0-100>, "summary": "<one sentence>"}},
        {{"rank": 5, "agent_name": "...", "score": <0-100>, "summary": "<one sentence>"}}
    ],
    "key_turning_points": [
        "<Round X: description of pivotal moment that changed game direction>",
        "<another turning point>"
    ],
    "reasoning_quality_comparison": "<detailed paragraph comparing the reasoning quality and sophistication across all agents>",
    "model_insights": "<observations about differences in agent capabilities, potential model differences, or notable patterns>"
}}
"""


# ============================================================================
# EVALUATOR CLASS
# ============================================================================

class QualitativeEvaluator:
    """
    LLM-as-a-Judge evaluator using G-Eval methodology.
    
    Implements academic standards for evaluation validity:
    - Explicit rubrics (G-Eval)
    - Chain-of-thought reasoning
    - Evidence-based scoring
    - Optional validation through multiple runs
    """
    
    def __init__(self, model: str = None, temperature: float = 0.3):
        """
        Initialize the evaluator.
        
        Args:
            model: Model to use for evaluation. Defaults to env var EVALUATOR_MODEL or LLM_MODEL.
            temperature: Lower temperature (0.3) for more consistent evaluation per G-Eval.
        """
        self.model = model or os.getenv("EVALUATOR_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))
        self.temperature = temperature
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
        )
        logger.info(f"QualitativeEvaluator initialized with model: {self.model}")
    
    def _get_role_specific_skill(self, role: str) -> str:
        """Get the role-specific skill to evaluate."""
        if role.lower() == "werewolf":
            return "deception_skill"
        else:
            return "detection_ability"
    
    def _build_rubrics_text(self, role: str) -> str:
        """Build the rubrics text for the evaluation prompt."""
        skills = ["reasoning_quality", "persuasion_effectiveness", 
                  self._get_role_specific_skill(role), "adaptation", "consistency"]
        
        rubrics_text = []
        for skill in skills:
            if skill in RUBRICS:
                rubrics_text.append(build_rubric_text(skill))
        
        return "\n\n".join(rubrics_text)
    
    def _extract_agent_data(
        self, 
        agent_name: str, 
        action_log: List[Dict], 
        debate_history: List[Dict]
    ) -> Tuple[str, str, str]:
        """Extract agent-specific data from game logs."""
        
        # Agent's actions with full reasoning
        agent_actions = [
            {
                "round": a.get("round"),
                "phase": a.get("phase"),
                "action": a.get("action"),
                "decision": a.get("decision"),
                "reasoning": a.get("reasoning"),
            }
            for a in action_log 
            if a.get("player") == agent_name
        ]
        actions_str = json.dumps(agent_actions, indent=2) if agent_actions else "No actions recorded"
        
        # Agent's debate statements
        agent_debates = [
            d.get("message", "") 
            for d in debate_history 
            if d.get("speaker") == agent_name
        ]
        debates_str = "\n".join([f"- {msg}" for msg in agent_debates]) if agent_debates else "No debate statements"
        
        # Agent's voting record
        agent_votes = [
            f"Round {a.get('round')}: voted for {a.get('decision')} (reason: {a.get('reasoning', 'N/A')[:100]}...)"
            for a in action_log 
            if a.get("player") == agent_name and a.get("action") == "vote"
        ]
        votes_str = "\n".join(agent_votes) if agent_votes else "No votes recorded"
        
        return actions_str, debates_str, votes_str
    
    def _calculate_weighted_score(self, skill_scores: Dict[str, SkillScore], role: str) -> float:
        """Calculate weighted overall score based on role-appropriate skills."""
        total_weight = 0
        weighted_sum = 0
        
        role_specific = self._get_role_specific_skill(role)
        
        for skill_name, skill_score in skill_scores.items():
            weight = RUBRICS.get(skill_name, {}).get("weight", 0.1)
            
            # Only count role-specific skill if it matches
            if skill_name in ["deception_skill", "detection_ability"]:
                if skill_name != role_specific:
                    continue
            
            weighted_sum += skill_score.score * 10 * weight  # Convert 1-10 to 10-100
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return 50.0
    
    async def evaluate_agent(
        self,
        agent_name: str,
        role: str,
        team: str,
        won: bool,
        survived: bool,
        roles: Dict[str, str],
        action_log: List[Dict],
        debate_history: List[Dict],
        winner: str,
        total_rounds: int,
    ) -> AgentEvaluation:
        """
        Evaluate a single agent using G-Eval methodology.
        """
        logger.info(f"Evaluating agent: {agent_name} ({role})")
        
        # Extract agent-specific data
        actions_str, debates_str, votes_str = self._extract_agent_data(
            agent_name, action_log, debate_history
        )
        
        # Build rubrics
        role_specific_skill = self._get_role_specific_skill(role)
        rubrics_text = self._build_rubrics_text(role)
        
        # Build prompt
        prompt = AGENT_EVALUATION_PROMPT.format(
            winner=winner,
            total_rounds=total_rounds,
            player_roles=", ".join([f"{n}: {r}" for n, r in roles.items()]),
            agent_name=agent_name,
            role=role,
            team=team,
            won=won,
            survived=survived,
            agent_actions=actions_str,
            debate_statements=debates_str,
            voting_history=votes_str,
            rubrics=rubrics_text,
            role_specific_skill=role_specific_skill,
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert game analyst evaluating AI agents. Respond ONLY with valid JSON, no markdown formatting."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2500,
            )
            
            content = response.choices[0].message.content
            content = self._clean_json_response(content)
            result = json.loads(content)
            
            # Build skill scores
            skill_scores = {}
            for skill_name, skill_data in result.get("skill_scores", {}).items():
                skill_scores[skill_name] = SkillScore(
                    skill_name=skill_name,
                    score=min(10, max(1, skill_data.get("score", 5))),  # Clamp 1-10
                    rubric_level=skill_data.get("rubric_level", "Average"),
                    evidence=skill_data.get("evidence", [])[:3],  # Limit evidence
                    explanation=skill_data.get("explanation", "")[:500],  # Limit length
                )
            
            # Calculate weighted score
            overall_score = result.get("overall_score")
            if overall_score is None or not (0 <= overall_score <= 100):
                overall_score = self._calculate_weighted_score(skill_scores, role)
            
            return AgentEvaluation(
                agent_name=agent_name,
                role=role,
                team=team,
                won=won,
                survived=survived,
                skill_scores=skill_scores,
                overall_score=overall_score,
                strengths=result.get("strengths", [])[:3],
                weaknesses=result.get("weaknesses", [])[:3],
                key_moments=result.get("key_moments", [])[:3],
                improvement_suggestions=result.get("improvement_suggestions", [])[:2],
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for agent {agent_name}: {e}")
            return self._default_evaluation(agent_name, role, team, won, survived)
        except Exception as e:
            logger.error(f"Error evaluating agent {agent_name}: {e}")
            return self._default_evaluation(agent_name, role, team, won, survived)
    
    def _default_evaluation(
        self, 
        agent_name: str, 
        role: str, 
        team: str, 
        won: bool, 
        survived: bool
    ) -> AgentEvaluation:
        """Return default evaluation when LLM evaluation fails."""
        return AgentEvaluation(
            agent_name=agent_name,
            role=role,
            team=team,
            won=won,
            survived=survived,
            overall_score=50.0,
            strengths=["Evaluation failed - using default"],
            weaknesses=["Unable to analyze due to evaluation error"],
        )
    
    async def compare_agents(
        self,
        agent_evaluations: List[AgentEvaluation],
        winner: str,
        total_rounds: int,
        action_log: List[Dict],
    ) -> Tuple[str, str, List[str], str, str, List[Dict]]:
        """
        Compare all agents and determine the best player.
        """
        logger.info("Running comparative evaluation...")
        
        # Build summaries
        summaries = []
        for eval in agent_evaluations:
            avg_skill = sum(s.score for s in eval.skill_scores.values()) / max(len(eval.skill_scores), 1)
            summaries.append({
                "agent_name": eval.agent_name,
                "role": eval.role,
                "team": eval.team,
                "won": eval.won,
                "overall_score": round(eval.overall_score, 1),
                "avg_skill_score": round(avg_skill, 1),
                "top_strengths": eval.strengths[:2],
                "main_weaknesses": eval.weaknesses[:2],
                "key_moments": eval.key_moments[:2],
            })
        
        # Build game events summary
        key_events = []
        for action in action_log:
            if action.get("action") in ["eliminate", "vote", "protect", "investigate"]:
                event = (
                    f"Round {action.get('round', '?')} ({action.get('phase', '?')}): "
                    f"{action.get('player', '?')} used {action.get('action', '?')} on {action.get('decision', '?')}"
                )
                if action.get("reasoning"):
                    event += f" - Reasoning: {action.get('reasoning', '')[:100]}..."
                key_events.append(event)
        
        prompt = COMPARATIVE_EVALUATION_PROMPT.format(
            winner=winner,
            total_rounds=total_rounds,
            player_list=", ".join([f"{e.agent_name} ({e.role})" for e in agent_evaluations]),
            individual_summaries=json.dumps(summaries, indent=2),
            game_events="\n".join(key_events[-20:]),
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert game analyst. Respond ONLY with valid JSON, no markdown formatting."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000,
            )
            
            content = response.choices[0].message.content
            content = self._clean_json_response(content)
            result = json.loads(content)
            
            # Update rankings in evaluations
            ranking = result.get("player_ranking", [])
            for rank_data in ranking:
                for eval in agent_evaluations:
                    if eval.agent_name == rank_data.get("agent_name"):
                        eval.rank = rank_data.get("rank", 0)
                        if "score" in rank_data:
                            eval.overall_score = rank_data.get("score", eval.overall_score)
            
            # Ensure all have ranks
            ranked_agents = {r.get("agent_name") for r in ranking}
            unranked = [e for e in agent_evaluations if e.agent_name not in ranked_agents]
            next_rank = len(ranking) + 1
            for eval in sorted(unranked, key=lambda x: -x.overall_score):
                eval.rank = next_rank
                next_rank += 1
            
            return (
                result.get("best_player", agent_evaluations[0].agent_name if agent_evaluations else "Unknown"),
                result.get("best_player_justification", "Unable to determine justification"),
                result.get("key_turning_points", []),
                result.get("reasoning_quality_comparison", ""),
                result.get("model_insights", ""),
                ranking,
            )
            
        except Exception as e:
            logger.error(f"Error in comparative evaluation: {e}")
            sorted_evals = sorted(agent_evaluations, key=lambda x: -x.overall_score)
            for i, eval in enumerate(sorted_evals):
                eval.rank = i + 1
            
            best = sorted_evals[0] if sorted_evals else None
            return (
                best.agent_name if best else "Unknown",
                f"{best.agent_name if best else 'Unknown'} had the highest overall score ({best.overall_score if best else 0:.1f}). Comparative evaluation failed, using quantitative fallback.",
                [],
                "Evaluation failed - using quantitative fallback",
                "",
                [],
            )
    
    async def validate_consistency(
        self,
        roles: Dict[str, str],
        action_log: List[Dict],
        debate_history: List[Dict],
        scores: List[Dict],
        winner: str,
        total_rounds: int,
        n_runs: int = 3,
    ) -> Dict[str, Any]:
        """
        Validate evaluation consistency by running multiple times.
        Academic standard: Inter-rater reliability (agreement rate) > 0.7
        """
        if n_runs < 2:
            return {"validated": False, "reason": "Need at least 2 runs for validation"}
        
        logger.info(f"Running validation with {n_runs} evaluation runs...")
        
        all_rankings = []
        all_best_players = []
        
        for run in range(n_runs):
            logger.info(f"Validation run {run + 1}/{n_runs}")
            
            evaluation = await self.evaluate_game(
                winner=winner,
                total_rounds=total_rounds,
                roles=roles,
                action_log=action_log,
                debate_history=debate_history,
                scores=scores,
                validate=False,
            )
            
            if evaluation:
                rankings = {e.agent_name: e.rank for e in evaluation.agent_evaluations}
                all_rankings.append(rankings)
                all_best_players.append(evaluation.best_player)
        
        if len(all_rankings) < 2:
            return {"validated": False, "reason": "Not enough successful runs"}
        
        # Calculate inter-rater reliability
        agreements = 0
        comparisons = 0
        
        agents = list(all_rankings[0].keys())
        for i in range(len(all_rankings)):
            for j in range(i + 1, len(all_rankings)):
                for agent in agents:
                    r1 = all_rankings[i].get(agent, 0)
                    r2 = all_rankings[j].get(agent, 0)
                    if r1 == r2:
                        agreements += 1
                    comparisons += 1
        
        agreement_rate = agreements / max(comparisons, 1)
        
        best_player_counts = {}
        for bp in all_best_players:
            best_player_counts[bp] = best_player_counts.get(bp, 0) + 1
        
        most_common_best = max(best_player_counts.items(), key=lambda x: x[1])
        best_player_consistency = most_common_best[1] / len(all_best_players)
        
        return {
            "validated": True,
            "n_runs": n_runs,
            "ranking_agreement_rate": round(agreement_rate, 3),
            "best_player_consistency": round(best_player_consistency, 3),
            "best_player_candidates": dict(best_player_counts),
            "is_reliable": agreement_rate > 0.6 and best_player_consistency > 0.5,
            "interpretation": (
                "Strong reliability" if agreement_rate > 0.8 else
                "Moderate reliability" if agreement_rate > 0.6 else
                "Weak reliability - results may vary between runs"
            ),
        }
    
    async def evaluate_game(
        self,
        winner: str,
        total_rounds: int,
        roles: Dict[str, str],
        action_log: List[Dict],
        debate_history: List[Dict],
        scores: List[Dict],
        validate: bool = False,
        validation_runs: int = 3,
    ) -> Optional[GameEvaluation]:
        """
        Complete game evaluation using LLM-as-Judge with G-Eval methodology.
        """
        try:
            logger.info(f"Starting qualitative evaluation (model: {self.model}, players: {len(roles)})")
            
            scores_by_player = {s.get("player_name"): s for s in scores}
            
            # Evaluate each agent
            agent_evaluations = []
            for player_name, role in roles.items():
                player_score = scores_by_player.get(player_name, {})
                team = "werewolves" if role.lower() == "werewolf" else "villagers"
                
                eval = await self.evaluate_agent(
                    agent_name=player_name,
                    role=role,
                    team=team,
                    won=player_score.get("won", team == winner),
                    survived=player_score.get("survived", False),
                    roles=roles,
                    action_log=action_log,
                    debate_history=debate_history,
                    winner=winner,
                    total_rounds=total_rounds,
                )
                agent_evaluations.append(eval)
            
            # Compare agents
            (
                best_player,
                justification,
                turning_points,
                reasoning_comparison,
                model_insights,
                ranking,
            ) = await self.compare_agents(
                agent_evaluations=agent_evaluations,
                winner=winner,
                total_rounds=total_rounds,
                action_log=action_log,
            )
            
            # Validation (optional)
            validation_result = {}
            if validate:
                validation_result = await self.validate_consistency(
                    roles=roles,
                    action_log=action_log,
                    debate_history=debate_history,
                    scores=scores,
                    winner=winner,
                    total_rounds=total_rounds,
                    n_runs=validation_runs,
                )
            
            logger.info(f"Evaluation complete. Best player: {best_player}")
            
            return GameEvaluation(
                game_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                winner=winner,
                total_rounds=total_rounds,
                best_player=best_player,
                best_player_justification=justification,
                agent_evaluations=agent_evaluations,
                key_turning_points=turning_points,
                reasoning_quality_comparison=reasoning_comparison,
                model_insights=model_insights,
                validation=validation_result,
            )
            
        except Exception as e:
            logger.exception(f"Error during game evaluation: {e}")
            return None
    
    def _clean_json_response(self, content: str) -> str:
        """Clean potential markdown formatting from JSON response."""
        content = content.strip()
        
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        
        return content.strip()


# ============================================================================
# CONVENIENCE FUNCTION FOR INTEGRATION
# ============================================================================

async def evaluate_game_quality(
    winner: str,
    total_rounds: int,
    roles: Dict[str, str],
    action_log: List[Dict[str, Any]],
    debate_history: List[Dict[str, str]],
    scores: List[Dict[str, Any]],
    validate: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to evaluate a game and return dict result.
    
    Usage in orchestrator:
        from green_agent.evaluator import evaluate_game_quality
        
        evaluation = await evaluate_game_quality(
            winner=self.winner,
            total_rounds=self.current_round,
            roles={n: r.value for n, r in self.roles.items()},
            action_log=self.action_log,
            debate_history=self.debate_history,
            scores=[s.model_dump() for s in scores],
            validate=False,
        )
        
        return AssessmentResult(..., evaluation=evaluation)
    
    Returns:
        Dict with evaluation results, or None if evaluation fails
    """
    evaluator = QualitativeEvaluator()
    evaluation = await evaluator.evaluate_game(
        winner=winner,
        total_rounds=total_rounds,
        roles=roles,
        action_log=action_log,
        debate_history=debate_history,
        scores=scores,
        validate=validate,
    )
    
    if evaluation:
        return evaluation.to_dict()
    return None
