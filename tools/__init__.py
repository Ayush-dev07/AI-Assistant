from tools.base import ToolRegistry, build_tool_definitions_for_llm


def register_all_tools() -> None:
    import tools.web
    import tools.api
    import tools.filesystem
    import tools.auto_skills
    import tools.code
    
    from tools.auto_skills.installer import SkillInstaller
    loaded = SkillInstaller().load_all_verified()
    if loaded:
        from core.logging import get_logger
        get_logger(__name__).info(
            "previously_installed_skills_loaded",
            count=len(loaded),
            names=loaded,
        )


__all__ = [
    "ToolRegistry",
    "build_tool_definitions_for_llm",
    "register_all_tools",
]