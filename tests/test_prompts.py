"""
Testes automatizados para validacao de prompts.
"""

import sys
from pathlib import Path

import pytest
import yaml

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure


def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestPrompts:
    PROMPTS_FILE = "prompts/bug_to_user_story_v2.yml"
    PROMPT_KEY = "bug_to_user_story_v2"

    def _get_prompt_data(self):
        prompts = load_prompts(self.PROMPTS_FILE)
        assert prompts is not None, f"Falha ao carregar {self.PROMPTS_FILE}"
        assert self.PROMPT_KEY in prompts, (
            f"Chave '{self.PROMPT_KEY}' nao encontrada em {self.PROMPTS_FILE}"
        )
        prompt_data = prompts[self.PROMPT_KEY]
        assert isinstance(prompt_data, dict), "Prompt deve ser um objeto (dict) no YAML."
        return prompt_data

    def _get_system_prompt(self):
        prompt_data = self._get_prompt_data()
        return str(prompt_data.get("system_prompt", ""))

    def test_prompt_has_system_prompt(self):
        """Verifica se o campo 'system_prompt' existe e nao esta vazio."""
        prompt_data = self._get_prompt_data()
        assert "system_prompt" in prompt_data, "Campo 'system_prompt' ausente."
        assert isinstance(prompt_data["system_prompt"], str), "'system_prompt' deve ser string."
        assert prompt_data["system_prompt"].strip(), "'system_prompt' nao pode estar vazio."

    def test_prompt_has_role_definition(self):
        """Verifica se o prompt define uma persona/role."""
        system_prompt = self._get_system_prompt().lower()
        role_markers = [
            "você é",
            "voce é",
            "assistente",
            "especialista",
            "product manager",
        ]
        assert any(marker in system_prompt for marker in role_markers), (
            "Prompt deve definir uma persona/role explicita (ex.: 'Voce e ...')."
        )

    def test_prompt_mentions_format(self):
        """Verifica se o prompt exige formato de User Story."""
        system_prompt = self._get_system_prompt().lower()
        format_markers = [
            "user story",
            "formato",
            "critérios de aceitação",
            "criterios de aceitacao",
            "como ",
            "eu quero",
            "para que",
        ]
        assert any(marker in system_prompt for marker in format_markers), (
            "Prompt deve exigir formato de saida (User Story + Criterios)."
        )

    def test_prompt_has_few_shot_examples(self):
        """Verifica se o prompt contem evidencias de exemplos (few-shot)."""
        system_prompt = self._get_system_prompt().lower()
        markers = [
            "[padrao",
            "retorne exatamente",
            "exemplo de cálculo",
            "exemplo de calculo",
        ]
        count = sum(1 for marker in markers if marker in system_prompt)
        assert count >= 2, "Prompt deve conter evidencias de few-shot/exemplos no texto."

    def test_prompt_no_todos(self):
        """Garante que nao ha TODOs no prompt."""
        prompt_data = self._get_prompt_data()
        all_text = yaml.safe_dump(prompt_data, allow_unicode=True).lower()
        assert "[todo]" not in all_text
        assert "todo:" not in all_text

    def test_minimum_techniques(self):
        """Verifica se pelo menos 2 tecnicas foram listadas."""
        prompt_data = self._get_prompt_data()

        # Valida regra formal ja existente no projeto.
        is_valid, errors = validate_prompt_structure(prompt_data)
        assert is_valid, f"Estrutura invalida: {errors}"

        techniques = prompt_data.get("techniques_applied", [])
        assert isinstance(techniques, list), "'techniques_applied' deve ser lista."
        assert len(techniques) >= 2, "Deve haver pelo menos 2 tecnicas aplicadas."


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
