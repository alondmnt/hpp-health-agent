"""Skill loading and parsing utilities.

Skills are markdown files that provide behavioral instructions to the LLM.
They are loaded from the skills/ directory and injected into prompts
at specific points (Orchestrator and Analyzer).

Usage:
    from pha.skill_loader import SkillLoader

    loader = SkillLoader()
    skills_text = loader.format_skills_for_prompt(["clinical-language-guard"], "analyzer")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional YAML support (graceful degradation if not installed)
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class SkillNotFoundError(Exception):
    """Raised when a requested skill file doesn't exist."""
    pass


@dataclass
class Skill:
    """Parsed skill with metadata and content."""

    name: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[Path] = None

    @property
    def version(self) -> str:
        return self.metadata.get("version", "1.0")

    @property
    def description(self) -> str:
        return self.metadata.get("description", "")

    @property
    def injection_points(self) -> List[str]:
        """Returns list of injection points. Defaults to orchestrator+analyzer if not specified."""
        points = self.metadata.get("injection_points", ["orchestrator", "analyzer"])
        valid_points = {"orchestrator", "analyzer"}
        return [p for p in points if p in valid_points] or ["orchestrator", "analyzer"]

    @property
    def tools(self) -> List[str]:
        """Returns list of relevant tools for this skill. Optional metadata field."""
        return self.metadata.get("tools", [])

    def applies_to(self, injection_point: str) -> bool:
        """Check if this skill applies to a given injection point."""
        return injection_point in self.injection_points


class SkillLoader:
    """Loads and parses skill files from the skills directory.

    Skills are markdown files with optional YAML frontmatter.

    Attributes:
        skills_dir: Path to the skills directory
        _cache: In-memory cache of loaded skills
    """

    # Default skills directory (repo_root/skills/)
    # __file__ = pha/skill_loader.py → parents[1] = repo_root
    DEFAULT_SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"

    def __init__(self, skills_dir: Optional[Path] = None):
        """Initialize the skill loader.

        Args:
            skills_dir: Path to skills directory. Defaults to repo_root/skills/
        """
        self.skills_dir = Path(skills_dir) if skills_dir else self.DEFAULT_SKILLS_DIR
        self._cache: Dict[str, Skill] = {}

        if not self.skills_dir.exists():
            print(f"Warning: Skills directory not found: {self.skills_dir}")

    def _parse_frontmatter(self, content: str) -> tuple[Dict[str, Any], str]:
        """Parse optional YAML frontmatter from markdown content.

        Args:
            content: Raw markdown file content

        Returns:
            Tuple of (metadata dict, remaining content)
        """
        if not content.startswith("---"):
            return {}, content

        lines = content.split("\n")
        end_index = None
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                end_index = i
                break

        if end_index is None:
            return {}, content

        frontmatter_lines = lines[1:end_index]
        frontmatter_text = "\n".join(frontmatter_lines)
        remaining_content = "\n".join(lines[end_index + 1:]).strip()

        metadata = {}
        if YAML_AVAILABLE and frontmatter_text.strip():
            try:
                metadata = yaml.safe_load(frontmatter_text) or {}
            except yaml.YAMLError as e:
                print(f"Warning: Failed to parse frontmatter YAML: {e}")
        elif frontmatter_text.strip():
            # Simple fallback parser for basic key: value pairs
            for line in frontmatter_lines:
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip()
                    if value.startswith("[") and value.endswith("]"):
                        value = [v.strip().strip('"\'') for v in value[1:-1].split(",")]
                    metadata[key] = value

        return metadata, remaining_content

    def _load_skill(self, skill_name: str) -> Skill:
        """Load a single skill by name.

        Args:
            skill_name: Name of the skill (without .md extension)

        Returns:
            Parsed Skill object

        Raises:
            SkillNotFoundError: If skill file doesn't exist
        """
        if skill_name in self._cache:
            return self._cache[skill_name]

        # Try both hyphen and underscore versions
        skill_path = self.skills_dir / f"{skill_name}.md"
        if not skill_path.exists():
            alt_path = self.skills_dir / f"{skill_name.replace('-', '_')}.md"
            if alt_path.exists():
                skill_path = alt_path
            else:
                # Also try underscore to hyphen
                alt_path2 = self.skills_dir / f"{skill_name.replace('_', '-')}.md"
                if alt_path2.exists():
                    skill_path = alt_path2
                else:
                    raise SkillNotFoundError(f"Skill '{skill_name}' not found at {skill_path}")

        raw_content = skill_path.read_text(encoding="utf-8")
        metadata, content = self._parse_frontmatter(raw_content)

        if "name" not in metadata:
            metadata["name"] = skill_name

        skill = Skill(
            name=skill_name,
            content=content,
            metadata=metadata,
            source_path=skill_path
        )

        self._cache[skill_name] = skill
        return skill

    def load_skills(
        self,
        skill_names: List[str],
        injection_point: Optional[str] = None
    ) -> List[Skill]:
        """Load multiple skills by name.

        Args:
            skill_names: List of skill names to load
            injection_point: If provided, filter to skills that apply to this point

        Returns:
            List of loaded Skill objects
        """
        skills = []
        for name in skill_names:
            try:
                skill = self._load_skill(name)
                if injection_point is None or skill.applies_to(injection_point):
                    skills.append(skill)
            except SkillNotFoundError as e:
                print(f"Warning: {e}")
        return skills

    def format_skills_for_prompt(
        self,
        skill_names: List[str],
        injection_point: str
    ) -> str:
        """Load skills and format them for prompt injection.

        Args:
            skill_names: List of skill names to load
            injection_point: Which injection point ("orchestrator" or "analyzer")

        Returns:
            Formatted string ready for prompt injection, or empty string if no skills
        """
        skills = self.load_skills(skill_names, injection_point=injection_point)

        if not skills:
            return ""

        lines = ["# ACTIVE SKILLS", ""]
        lines.append("The following behavioral skills are active for this request:")
        lines.append("")

        for skill in skills:
            lines.append(f"## Skill: {skill.name}")
            if skill.description:
                lines.append(f"*{skill.description}*")
            lines.append("")
            lines.append(skill.content)
            lines.append("")

        return "\n".join(lines)

    def list_available_skills(self) -> List[str]:
        """List all available skill names in the skills directory.

        Returns:
            List of skill names (without .md extension)
        """
        if not self.skills_dir.exists():
            return []

        return [p.stem for p in self.skills_dir.glob("*.md")]

    def clear_cache(self) -> None:
        """Clear the skill cache."""
        self._cache.clear()
