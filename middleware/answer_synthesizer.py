"""
middleware/answer_synthesizer.py

Generates the final natural language answer using the synthesis LLM.

Uses llama3.2:3b for all synthesis. The prompt is carefully designed
to instruct the LLM to reason over multi-row results, make comparisons,
and answer yes/no questions directly from the data.
"""

from __future__ import annotations

import requests
import yaml

from pathlib import Path

from middleware.models import DBResult


def _load_model_config() -> dict:
    """Load model configuration from intents.yaml.

    Returns:
        dict: The 'models' section of the config.
    """
    config_path = Path(__file__).parent.parent / "config" / "intents.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["models"]


def synthesize_answer(
    user_question: str,
    context: str,
    db_result: DBResult,
) -> str:
    """Generate a natural language answer from database context.

    If self-healing was triggered, the context IS the answer.
    For normal results, calls llama3.2:3b with a reasoning-aware
    prompt that handles comparisons, rankings, and yes/no questions.

    Args:
        user_question: The original user question.
        context: Formatted context string from context_formatter.
        db_result: The DBResult (used to check self-healing flag).

    Returns:
        str: The final answer to show the user.
    """
    # Self-healing case — context is already the follow-up message
    if db_result.self_healing_triggered:
        return context

    model_cfg = _load_model_config()
    synthesis_model: str = model_cfg["synthesis_model"]
    base_url: str        = model_cfg["ollama_base_url"]
    temperature: float   = model_cfg["synthesis_temperature"]
    max_tokens: int      = model_cfg["max_tokens_synthesis"]

    prompt = f"""You are a helpful data assistant for a company's internal system.
Answer the user's question using ONLY the information in the database results below.

STRICT RULES:
1. Use ONLY the provided data — never invent or assume facts not shown.
2. Answer the question DIRECTLY and concisely first, then add supporting details.
3. Format currency with $ and commas (e.g. $95,000.00).
4. Do NOT use technical words like "database", "query", "records", "context".
5. Write naturally as a knowledgeable assistant.

REASONING RULES — follow these for specific question types:
- COMPARISON questions ("highest", "lowest", "most", "least", "best"):
    Look through ALL records provided, identify the answer, state it clearly.
    Example: If asked "who earns the most?" scan all salaries and name the highest.
- YES/NO questions ("is X the highest?", "does X have Y?", "are there any?"):
    Answer YES or NO first, then explain using the data.
    Example: "No, Alan Turing ($115,000) is not the highest paid.
    Linus Torvalds earns the most at $130,000."
- RANKING questions ("rank by", "order by", "top 5"):
    Present results as a numbered list in the order given.
- PARTIAL DATA questions (when data only partly answers the question):
    Answer what you can, then clearly state what information isn't available.

{context}

User question: {user_question}

Answer:"""

    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": synthesis_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()

        if not answer:
            return "I found the data but had trouble forming a response. Please try again."

        return answer

    except requests.RequestException:
        return (
            "I retrieved the data but couldn't generate a response right now. "
            f"Here is what I found:\n\n{context}"
        )
