from dataclasses import dataclass, field


@dataclass
class DefenseResult:
    allowed: bool  # True = request passed defense, False = blocked
    modified_input: str | None = None  # If defense modified the input
    modified_output: str | None = None  # If defense modified the output
    flags: list[str] = field(default_factory=list)  # Warning flags raised
    confidence: float = 1.0  # How confident the defense is in its decision


class Defense:
    name: str = "base"

    def check_input(self, user_query: str, context: list[dict] | None = None) -> DefenseResult:
        """Check input before it reaches the agent."""
        return DefenseResult(allowed=True)

    def check_output(self, agent_output: str, recommendation: dict | None = None) -> DefenseResult:
        """Check output after agent produces it."""
        return DefenseResult(allowed=True)
