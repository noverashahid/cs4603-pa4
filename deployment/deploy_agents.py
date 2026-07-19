"""Bonus B — deploy via the databricks-agents SDK (deployment/deploy_agents.py)."""

from __future__ import annotations

import os

import mlflow

from config import get_settings
from deployment.deploy import CODE_PATHS, MODEL_NAME as _PART2_MODEL_NAME, PIP_REQUIREMENTS, ROOT

MODEL_NAME = "pa4_document_analyst"

CHAT_MODEL_PATH = os.path.join(ROOT, "deployment", "agent_model.py")

INPUT_EXAMPLE = {"messages": [{"role": "user", "content": "What was the revenue?"}]}


def log_and_register_for_agents():
    """Log the ChatModel-wrapped graph and register it in Unity Catalog.

    Declares BOTH the Vector Search index and the LLM serving endpoint as
    explicit resource dependencies so agents.deploy() can automatically
    provision (and rotate) short-lived OBO credentials for the endpoint's
    service principal — without this, the endpoint's own identity has no
    permission on those resources and every call fails with
    PermissionDenied: Invalid access token.

    This only works end-to-end because config.get_chat_llm() now uses
    databricks_langchain.ChatDatabricks (auto-auth aware) instead of
    langchain_openai.ChatOpenAI + an explicit PAT, and because
    environment_vars below no longer injects DATABRICKS_HOST/TOKEN — an
    explicit PAT in the container's env would otherwise take precedence
    over the auto-provisioned credential in the SDK's default auth chain.
    """
    from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex

    settings = get_settings()

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(f"/Shared/{MODEL_NAME}_agents_experiment")

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=CHAT_MODEL_PATH,
            name="agent_chat",
            code_paths=CODE_PATHS,
            pip_requirements=PIP_REQUIREMENTS,
            input_example=INPUT_EXAMPLE,
            resources=[
                DatabricksVectorSearchIndex(index_name=settings["vs_index"]),
                DatabricksServingEndpoint(endpoint_name=settings["model"]),
            ],
        )

    catalog = os.environ["UC_CATALOG"]
    schema = os.environ["UC_SCHEMA"]
    uc_name = f"{catalog}.{schema}.{MODEL_NAME}_chat"

    registered = mlflow.register_model(model_info.model_uri, uc_name)
    print(f"Registered model: {uc_name}, version {registered.version}")

    return uc_name, registered.version


def main() -> None:
    from databricks import agents

    settings = get_settings()
    uc_name, version = log_and_register_for_agents()

    print(f"Deploying {uc_name} version {version} via agents.deploy()...")

    deployment = agents.deploy(
        model_name=uc_name,
        model_version=version,
        scale_to_zero=True,
        endpoint_name="pa4-document-analyst-review",
        environment_vars={
            # Serving endpoint name for ChatDatabricks — not a credential.
            # DATABRICKS_HOST/DATABRICKS_TOKEN are deliberately NOT set here:
            # agents.deploy() auto-provisions OBO credentials for the
            # resources declared in log_and_register_for_agents(); an
            # explicit PAT would take precedence over that in the SDK's
            # default auth chain and defeat automatic authentication.
            "DATABRICKS_MODEL": "{{secrets/cs4603-deploy/DATABRICKS_MODEL}}",
            # RAG / Vector Search variables
            "VECTOR_SEARCH_ENDPOINT": settings["vs_endpoint"],
            "VECTOR_SEARCH_INDEX": settings["vs_index"],
            "EMBEDDINGS_ENDPOINT": settings["embeddings"],
        }
    )

    print("\nDeployment complete.")
    print(f"Endpoint name: {deployment.endpoint_name}")
    print(f"Review App URL: {deployment.review_app_url}")


if __name__ == "__main__":
    main()