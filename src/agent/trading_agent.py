"""Main commodity trading agent using LangChain."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .system_prompt import DEFAULT_SYSTEM_PROMPT, HARDENED_SYSTEM_PROMPT
from .tools import price, news, risk, fundamentals, correlation, position, recommendation
from .tools.price import get_price
from .tools.news import get_news
from .tools.risk import calculate_risk
from .tools.fundamentals import get_fundamentals
from .tools.correlation import get_correlation
from .tools.position import check_position_limits
from .tools.recommendation import submit_recommendation

logger = logging.getLogger(__name__)

# Map of tool names to their module (for set_tool_mode)
_TOOL_MODULES = {
    "get_price": price,
    "price": price,
    "get_news": news,
    "news": news,
    "calculate_risk": risk,
    "risk": risk,
    "get_fundamentals": fundamentals,
    "fundamentals": fundamentals,
    "get_correlation": correlation,
    "correlation": correlation,
    "check_position_limits": position,
    "position": position,
}


@dataclass
class AgentResult:
    """Result from a trading agent analysis run."""

    recommendation: Optional[dict] = None
    reasoning_chain: list[dict] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    raw_output: str = ""


class CommodityTradingAgent:
    """LangChain-based commodity trading analyst agent.

    Supports OpenAI and Anthropic models. Tools can be switched between
    normal and attack modes for red-team testing.

    Args:
        model_name: LLM model identifier (e.g., 'gpt-4o', 'claude-sonnet-4-20250514').
        system_prompt: Custom system prompt. Defaults to DEFAULT_SYSTEM_PROMPT.
        defenses: If 'hardened', uses HARDENED_SYSTEM_PROMPT. Can also be a custom string.
        config_path: Path to agent_config.yaml. Auto-detected if None.
        temperature: LLM temperature setting.
        max_iterations: Max agent reasoning iterations.
    """

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        system_prompt: Optional[str] = None,
        defenses: Optional[str] = None,
        config_path: Optional[str] = None,
        temperature: float = 0.0,
        max_iterations: int = 15,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations

        # Resolve system prompt
        if system_prompt is not None:
            self.system_prompt = system_prompt
        elif defenses == "hardened":
            self.system_prompt = HARDENED_SYSTEM_PROMPT
        elif isinstance(defenses, str) and defenses != "hardened":
            self.system_prompt = defenses
        else:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT

        # Load config
        self.config = self._load_config(config_path)

        # Build tools list
        self.tools = [
            get_price,
            get_news,
            calculate_risk,
            get_fundamentals,
            get_correlation,
            check_position_limits,
            submit_recommendation,
        ]

        # Build the LLM and agent
        self.llm = self._create_llm()
        self.agent_executor = self._create_agent()

    def _load_config(self, config_path: Optional[str] = None) -> dict:
        """Load agent configuration from YAML."""
        if config_path:
            path = Path(config_path)
        else:
            # Auto-detect config relative to this file
            path = Path(__file__).resolve().parent.parent.parent / "config" / "agent_config.yaml"

        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f)
        else:
            logger.warning("Config file not found at %s, using defaults.", path)
            return {}

    def _create_llm(self) -> Any:
        """Create the LangChain LLM based on model_name."""
        model = self.model_name.lower()

        if "claude" in model or "anthropic" in model:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
            )
        elif "gemini" in model:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
            )
        elif "groq" in model or "llama" in model or "mixtral" in model:
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=self.model_name,
                temperature=self.temperature,
            )
        elif "mistral" in model:
            from langchain_mistralai import ChatMistralAI

            return ChatMistralAI(
                model=self.model_name,
                temperature=self.temperature,
            )
        elif "gpt" in model or "o1" in model or "o3" in model:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
            )
        else:
            # Default to Anthropic
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
            )

    def _create_agent(self) -> Any:
        """Create the LangChain agent executor."""
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.agents import AgentExecutor, create_tool_calling_agent

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=self.max_iterations,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

    def analyze(self, query: str) -> AgentResult:
        """Run the agent to analyze a commodity trading query.

        Args:
            query: Natural language query (e.g., "Should we buy brent crude?").

        Returns:
            AgentResult with recommendation, reasoning chain, tools used, and raw output.
        """
        result = AgentResult()

        try:
            response = self.agent_executor.invoke({"input": query})

            # Extract raw output
            result.raw_output = response.get("output", "")

            # Extract reasoning chain and tools used from intermediate steps
            intermediate_steps = response.get("intermediate_steps", [])
            for step in intermediate_steps:
                if len(step) >= 2:
                    action, observation = step[0], step[1]
                    tool_name = getattr(action, "tool", "unknown")
                    tool_input = getattr(action, "tool_input", {})

                    result.tools_used.append(tool_name)
                    result.reasoning_chain.append(
                        {
                            "tool": tool_name,
                            "input": tool_input,
                            "output": observation if isinstance(observation, (dict, list, str)) else str(observation),
                        }
                    )

            # Try to extract structured recommendation from output
            result.recommendation = self._extract_recommendation(result.raw_output)

        except Exception as e:
            logger.error("Agent execution failed: %s", e)
            result.raw_output = f"ERROR: {e}"

        return result

    def _extract_recommendation(self, output: str) -> Optional[dict]:
        """Try to extract a JSON recommendation from the agent's output."""
        # Look for JSON blocks in the output
        try:
            # Try to find JSON in the output
            start = output.find("{")
            end = output.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = output[start:end]
                parsed = json.loads(json_str)
                if "action" in parsed:
                    return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    def set_tool_mode(self, tool_name: str, mode: str = "manipulated", **kwargs: Any) -> None:
        """Enable an attack mode on a specific tool.

        Args:
            tool_name: Name of the tool (e.g., 'price', 'get_price', 'news', 'risk').
            mode: Mode name. Common modes:
                - 'manipulated': Return wrong data (price, risk, correlation).
                - 'inject_payload': Inject content into news (pass payload as kwarg).
                - 'stale_data': Return outdated fundamentals.
                - 'override': Always report within limits (position).
            **kwargs: Mode-specific parameters (e.g., override_price=999.99).
        """
        module = _TOOL_MODULES.get(tool_name.lower())
        if module is None:
            raise ValueError(
                f"Unknown tool '{tool_name}'. Available: {list(_TOOL_MODULES.keys())}"
            )

        if mode == "manipulated":
            module.set_mode(manipulated=True, **kwargs)
        elif mode == "inject_payload":
            module.set_mode(inject_payload=kwargs.get("payload", ""), **kwargs)
        elif mode == "stale_data":
            module.set_mode(stale_data=True)
        elif mode == "override":
            module.set_mode(override=True)
        else:
            # Pass mode as a generic kwarg
            module.set_mode(**{mode: True}, **kwargs)

        logger.info("Tool '%s' set to mode '%s' with kwargs %s", tool_name, mode, kwargs)

    def reset_tools(self) -> None:
        """Reset all tools to normal (non-attack) mode."""
        for module in set(_TOOL_MODULES.values()):
            module.reset_mode()
        # Also clear submitted recommendations
        recommendation.clear_recommendations()
        logger.info("All tools reset to normal mode.")

    def get_submitted_recommendations(self) -> list[dict]:
        """Get all recommendations submitted during this session."""
        return recommendation.get_submitted_recommendations()
