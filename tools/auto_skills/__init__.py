from tools.base import ToolRegistry
from tools.auto_skills.installer import SkillInstaller, InstallRecord
from tools.auto_skills.skill_tool import SkillInstallTool
from tools.auto_skills.verifier import SkillVerifier, VerificationResult

# Auto-register the install_skill meta-tool on import
ToolRegistry.register(SkillInstallTool())

__all__ = [
    "SkillVerifier",
    "VerificationResult",
    "SkillInstaller",
    "InstallRecord",
    "SkillInstallTool",
]