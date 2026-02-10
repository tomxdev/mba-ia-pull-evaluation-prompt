"""
Script de avaliacao de prompts no LangSmith.

Fase 5:
- Usa as 4 metricas especificas do desafio:
  1) Tone Score
  2) Acceptance Criteria Score
  3) User Story Format Score
  4) Completeness Score
- Regra de aprovacao:
  - cada metrica >= 0.9
  - media das 4 >= 0.9
"""

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langsmith import Client
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

from utils import check_env_vars, format_score, print_section_header, get_llm as get_configured_llm
from metrics import (
    evaluate_tone_score,
    evaluate_acceptance_criteria_score,
    evaluate_user_story_format_score,
    evaluate_completeness_score,
)

load_dotenv()

DATASET_JSONL_PATH = "datasets/bug_to_user_story.jsonl"


def get_llm():
    return get_configured_llm(temperature=0)


def resolve_prompt_name(prompt_name: str) -> str:
    """Resolve nome no formato owner/prompt quando USERNAME_LANGSMITH_HUB estiver definido."""
    if "/" in prompt_name:
        return prompt_name
    username = os.getenv("USERNAME_LANGSMITH_HUB", "").strip()
    if username:
        return f"{username}/{prompt_name}"
    return prompt_name


def parse_prompts_to_evaluate() -> List[str]:
    raw = os.getenv("PROMPTS_TO_EVALUATE", "bug_to_user_story_v1,bug_to_user_story_v2")
    prompts = [p.strip() for p in raw.split(",") if p.strip()]
    return prompts


def load_dataset_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples
    except FileNotFoundError:
        print(f"ERRO: Arquivo nao encontrado: {jsonl_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"ERRO: JSONL invalido: {e}")
        return []
    except Exception as e:
        print(f"ERRO: Falha ao carregar dataset: {e}")
        return []


def create_evaluation_dataset(client: Client, dataset_name: str, jsonl_path: str) -> str:
    print(f"Criando dataset de avaliacao: {dataset_name}...")

    examples = load_dataset_from_jsonl(jsonl_path)
    if not examples:
        print("ERRO: Nenhum exemplo carregado do JSONL")
        return dataset_name

    print(f"   OK: Carregados {len(examples)} exemplos do arquivo {jsonl_path}")

    try:
        datasets = client.list_datasets(dataset_name=dataset_name)
        existing_dataset = None
        for ds in datasets:
            if ds.name == dataset_name:
                existing_dataset = ds
                break

        if existing_dataset:
            print(f"   OK: Dataset '{dataset_name}' ja existe, usando existente")
            return dataset_name

        dataset = client.create_dataset(dataset_name=dataset_name)
        for example in examples:
            client.create_example(
                dataset_id=dataset.id,
                inputs=example["inputs"],
                outputs=example["outputs"],
            )

        print(f"   OK: Dataset criado com {len(examples)} exemplos")
        return dataset_name

    except Exception as e:
        print(f"   AVISO: Erro ao criar dataset: {e}")
        return dataset_name


def pull_prompt_from_langsmith(prompt_name: str) -> ChatPromptTemplate:
    resolved_name = resolve_prompt_name(prompt_name)
    try:
        print(f"   Puxando prompt do LangSmith Hub: {resolved_name}")
        prompt = hub.pull(resolved_name)
        print("   OK: Prompt carregado com sucesso")
        return prompt
    except Exception as e:
        # Fallback para baseline publico do desafio.
        if prompt_name == "bug_to_user_story_v1" and resolved_name != "leonanluppi/bug_to_user_story_v1":
            try:
                fallback_name = "leonanluppi/bug_to_user_story_v1"
                print(f"   AVISO: prompt nao encontrado em {resolved_name}, tentando fallback {fallback_name}")
                prompt = hub.pull(fallback_name)
                print("   OK: Prompt baseline carregado com sucesso")
                return prompt
            except Exception:
                pass

        print("\n" + "=" * 70)
        print(f"ERRO: Nao foi possivel carregar o prompt '{resolved_name}'")
        print("=" * 70)
        print(str(e))
        print("\nAcoes sugeridas:")
        print("1. Rode: python src/push_prompts.py")
        print("2. Confirme no Hub se o prompt foi publicado")
        print("3. Verifique USERNAME_LANGSMITH_HUB no .env")
        print("=" * 70 + "\n")
        raise


def evaluate_prompt_on_example(
    prompt_template: ChatPromptTemplate,
    example: Any,
    llm: Any,
) -> Dict[str, str]:
    try:
        if hasattr(example, "inputs") and hasattr(example, "outputs"):
            inputs = example.inputs
            outputs = example.outputs
        elif isinstance(example, dict):
            inputs = example.get("inputs", {})
            outputs = example.get("outputs", {})
        else:
            inputs = {}
            outputs = {}

        chain = prompt_template | llm
        response = chain.invoke(inputs)
        answer = response.content

        bug_report = inputs.get("bug_report", "") if isinstance(inputs, dict) else ""
        reference = outputs.get("reference", "") if isinstance(outputs, dict) else ""

        return {
            "bug_report": bug_report,
            "answer": answer,
            "reference": reference,
        }
    except Exception as e:
        print(f"      AVISO: Erro ao avaliar exemplo: {e}")
        return {"bug_report": "", "answer": "", "reference": ""}


def evaluate_prompt(prompt_name: str, dataset_name: str, client: Client) -> Dict[str, float]:
    print(f"\nAvaliando: {prompt_name}")

    try:
        prompt_template = pull_prompt_from_langsmith(prompt_name)

        examples = load_dataset_from_jsonl(DATASET_JSONL_PATH)
        if not examples:
            examples = list(client.list_examples(dataset_name=dataset_name))

        print(f"   Dataset: {len(examples)} exemplos")
        llm = get_llm()

        max_examples = int(os.getenv("EVAL_MAX_EXAMPLES", "10"))
        batch = examples[:max_examples]

        tone_scores: List[float] = []
        acceptance_scores: List[float] = []
        format_scores: List[float] = []
        completeness_scores: List[float] = []

        print("   Avaliando exemplos...")

        for i, example in enumerate(batch, 1):
            result = evaluate_prompt_on_example(prompt_template, example, llm)
            if not result["answer"]:
                continue

            tone = evaluate_tone_score(result["bug_report"], result["answer"], result["reference"])
            acceptance = evaluate_acceptance_criteria_score(
                result["bug_report"], result["answer"], result["reference"]
            )
            story_format = evaluate_user_story_format_score(
                result["bug_report"], result["answer"], result["reference"]
            )
            completeness = evaluate_completeness_score(
                result["bug_report"], result["answer"], result["reference"]
            )

            tone_scores.append(tone["score"])
            acceptance_scores.append(acceptance["score"])
            format_scores.append(story_format["score"])
            completeness_scores.append(completeness["score"])

            print(
                f"      [{i}/{len(batch)}] "
                f"Tone:{tone['score']:.2f} "
                f"Acc:{acceptance['score']:.2f} "
                f"Format:{story_format['score']:.2f} "
                f"Comp:{completeness['score']:.2f}"
            )

        avg_tone = sum(tone_scores) / len(tone_scores) if tone_scores else 0.0
        avg_acceptance = sum(acceptance_scores) / len(acceptance_scores) if acceptance_scores else 0.0
        avg_format = sum(format_scores) / len(format_scores) if format_scores else 0.0
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
        avg_specific = (avg_tone + avg_acceptance + avg_format + avg_completeness) / 4

        return {
            "tone_score": round(avg_tone, 4),
            "acceptance_criteria_score": round(avg_acceptance, 4),
            "user_story_format_score": round(avg_format, 4),
            "completeness_score": round(avg_completeness, 4),
            "average_specific": round(avg_specific, 4),
        }

    except Exception as e:
        print(f"   ERRO na avaliacao: {e}")
        return {
            "tone_score": 0.0,
            "acceptance_criteria_score": 0.0,
            "user_story_format_score": 0.0,
            "completeness_score": 0.0,
            "average_specific": 0.0,
        }


def display_results(prompt_name: str, scores: Dict[str, float]) -> bool:
    print("\n" + "=" * 50)
    print(f"Prompt: {resolve_prompt_name(prompt_name)}")
    print("=" * 50)

    print("\nMetricas especificas (desafio):")
    print(f"  - Tone Score: {format_score(scores['tone_score'], threshold=0.9)}")
    print(
        f"  - Acceptance Criteria Score: "
        f"{format_score(scores['acceptance_criteria_score'], threshold=0.9)}"
    )
    print(
        f"  - User Story Format Score: "
        f"{format_score(scores['user_story_format_score'], threshold=0.9)}"
    )
    print(f"  - Completeness Score: {format_score(scores['completeness_score'], threshold=0.9)}")

    average_score = scores["average_specific"]

    print("\n" + "-" * 50)
    print(f"MEDIA DAS 4 METRICAS: {average_score:.4f}")
    print("-" * 50)

    metricas_4 = [
        scores["tone_score"],
        scores["acceptance_criteria_score"],
        scores["user_story_format_score"],
        scores["completeness_score"],
    ]
    passed = all(m >= 0.9 for m in metricas_4) and average_score >= 0.9

    if passed:
        print("\nSTATUS: APROVADO")
        print("Regra: todas as 4 metricas >= 0.9 e media >= 0.9")
    else:
        print("\nSTATUS: REPROVADO")
        print("Regra: alguma metrica < 0.9 ou media < 0.9")

    return passed


def main() -> int:
    print_section_header("AVALIACAO DE PROMPTS OTIMIZADOS")

    provider = os.getenv("LLM_PROVIDER", "openai")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    eval_model = os.getenv("EVAL_MODEL", "gpt-4o")

    print(f"Provider: {provider}")
    print(f"Modelo Principal: {llm_model}")
    print(f"Modelo de Avaliacao: {eval_model}\n")

    required_vars = ["LANGSMITH_API_KEY", "LLM_PROVIDER"]
    if provider == "openai":
        required_vars.append("OPENAI_API_KEY")
    elif provider in ["google", "gemini"]:
        required_vars.append("GOOGLE_API_KEY")

    if not check_env_vars(required_vars):
        return 1

    client = Client()
    project_name = os.getenv("LANGCHAIN_PROJECT", "prompt-optimization-challenge-resolved")

    if not Path(DATASET_JSONL_PATH).exists():
        print(f"ERRO: Dataset nao encontrado: {DATASET_JSONL_PATH}")
        return 1

    dataset_name = f"{project_name}-eval"
    create_evaluation_dataset(client, dataset_name, DATASET_JSONL_PATH)

    print("\n" + "=" * 70)
    print("PROMPTS PARA AVALIAR")
    print("=" * 70)
    print("\nEste script puxa prompts do LangSmith Hub.")
    print("Certifique-se de ter feito push dos prompts antes de avaliar:")
    print("  python src/push_prompts.py\n")

    prompts_to_evaluate = parse_prompts_to_evaluate()

    all_passed = True
    results_summary = []

    for prompt_name in prompts_to_evaluate:
        scores = evaluate_prompt(prompt_name, dataset_name, client)
        passed = display_results(prompt_name, scores)
        all_passed = all_passed and passed
        results_summary.append({"prompt": prompt_name, "scores": scores, "passed": passed})

    print("\n" + "=" * 50)
    print("RESUMO FINAL")
    print("=" * 50 + "\n")

    print(f"Prompts avaliados: {len(results_summary)}")
    print(f"Aprovados: {sum(1 for r in results_summary if r['passed'])}")
    print(f"Reprovados: {sum(1 for r in results_summary if not r['passed'])}\n")

    if all_passed:
        print("Todos os prompts atingiram o criterio da Fase 5.")
        print(f"Projeto: https://smith.langchain.com/projects/{project_name}")
        return 0

    print("Alguns prompts nao atingiram o criterio da Fase 5.")
    print("Proximos passos:")
    print("1. Refatorar prompt com score baixo")
    print("2. Rodar: python src/push_prompts.py")
    print("3. Rodar: python src/evaluate.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
