"""
Script para fazer push de prompts ao LangSmith Prompt Hub.

Padrao:
- Le prompts de prompts/bug_to_user_story_v2.yml
- Publica cada prompt no Hub (publico)

Opcional no .env:
- PROMPTS_FILE=prompts/bug_to_user_story_v2.yml
- USERNAME_LANGSMITH_HUB=seu_usuario
"""

import os
import sys
from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, print_section_header, validate_prompt_structure

load_dotenv()


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """Faz push de um prompt para o LangSmith Hub."""
    try:
        username = os.getenv("USERNAME_LANGSMITH_HUB", "").strip()
        full_prompt_name = prompt_name if "/" in prompt_name else (
            f"{username}/{prompt_name}" if username else prompt_name
        )

        system_prompt = (prompt_data.get("system_prompt") or "").strip()
        user_prompt = (prompt_data.get("user_prompt") or "{bug_report}").strip()

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", user_prompt),
            ]
        )

        description = (prompt_data.get("description") or "").strip() or None
        tags = prompt_data.get("tags")
        techniques = prompt_data.get("techniques_applied", [])
        if not isinstance(tags, list):
            tags = []
        if isinstance(techniques, list):
            tags = list(dict.fromkeys(tags + techniques))

        readme_parts = []
        if description:
            readme_parts.append(f"Description: {description}")
        if isinstance(techniques, list) and techniques:
            readme_parts.append("Techniques: " + ", ".join(techniques))
        readme = "\n".join(readme_parts) if readme_parts else None

        result = hub.push(
            full_prompt_name,
            prompt_template,
            new_repo_is_public=True,
            new_repo_description=description,
            readme=readme,
            tags=tags or None,
        )

        print(f"   OK: {full_prompt_name}")
        print(f"   Hub response: {result}")
        return True
    except Exception as e:
        message = str(e).lower()
        if "409" in message and "nothing to commit" in message:
            print(f"   OK: {prompt_name} ja publicado (sem alteracoes para commit).")
            return True
        print(f"   ERRO no push de '{prompt_name}': {e}")
        return False


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """Valida estrutura basica de um prompt."""
    return validate_prompt_structure(prompt_data)


def main():
    """Funcao principal."""
    print_section_header("PUSH DE PROMPTS PARA O LANGSMITH HUB")

    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        print("ERRO: defina LANGSMITH_API_KEY (ou LANGCHAIN_API_KEY) no .env")
        return 1

    prompt_file = os.getenv("PROMPTS_FILE", "prompts/bug_to_user_story_v2.yml")

    if not os.path.exists(prompt_file):
        print(f"ERRO: arquivo nao encontrado: {prompt_file}")
        return 1

    prompt_config = load_yaml(prompt_file)
    if not prompt_config or not isinstance(prompt_config, dict):
        print("ERRO: arquivo de prompts invalido ou vazio")
        return 1

    print(f"Arquivo de prompts: {prompt_file}")

    username = os.getenv("USERNAME_LANGSMITH_HUB", "").strip()
    if username:
        print(f"Owner no Hub: {username}")
    else:
        print("Aviso: USERNAME_LANGSMITH_HUB vazio. Push sem owner explicito.")

    ok_count = 0
    fail_count = 0

    for prompt_name, prompt_data in prompt_config.items():
        print(f"\nPublicando prompt: {prompt_name}")

        is_valid, errors = validate_prompt(prompt_data or {})
        if not is_valid:
            fail_count += 1
            print("   ERRO de validacao:")
            for err in errors:
                print(f"   - {err}")
            continue

        if push_prompt_to_langsmith(prompt_name, prompt_data):
            ok_count += 1
        else:
            fail_count += 1

    print("\nResumo:")
    print(f"- Publicados: {ok_count}")
    print(f"- Falhas: {fail_count}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
