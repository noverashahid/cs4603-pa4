"""Log, register, and serve the Document Analyst (Tasks 2.2 + 2.3).

Run:  uv run python deployment/deploy.py

TODO:
  - `log_and_register()`: set registry uri to 'databricks-uc', log the model via
    `mlflow.langchain.log_model(lc_model="deployment/agent_model.py", name=...,
    code_paths=[...], pip_requirements=[...], input_example={...})`, then
    `mlflow.register_model(...)` into $UC_CATALOG.$UC_SCHEMA.<model>.
  - `create_or_update_endpoint(uc_name, version)`: create/update a Model Serving
    endpoint with `WorkspaceClient().serving_endpoints`, workload_size='Small',
    scale_to_zero_enabled=True, and environment_vars supplied as secret refs
    ({{secrets/cs4603-deploy/...}}). Wait for READY and print the URL.
"""

from __future__ import annotations

import os

import mlflow

from config import get_settings

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_MODEL_PATH = os.path.join(ROOT, "deployment", "agent_model.py")

MODEL_NAME = "pa4_document_analyst"
ENDPOINT_NAME = os.environ.get("SERVING_ENDPOINT_NAME", "pa4-document-analyst")
SECRET_SCOPE = "cs4603-deploy"

PIP_REQUIREMENTS = [
    "mlflow>=2.16.0",
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-openai>=0.2.0",
    "databricks-langchain>=0.1.0",
    "databricks-vectorsearch>=0.40",
    "databricks-sdk>=0.23.0",
    "langchain-mcp-adapters>=0.0.5",
    "mcp>=1.0.0",
    "openai>=1.40.0",
    "python-dotenv>=1.0.0",
]

CODE_PATHS = [
    os.path.join(ROOT, "agent"),
    os.path.join(ROOT, "rag"),
    os.path.join(ROOT, "tools"),
    os.path.join(ROOT, "config.py"),
]

INPUT_EXAMPLE = {"messages": [{"role": "user", "content": "What was the revenue?"}]}


def log_and_register():
    settings = get_settings()

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(f"/Shared/{MODEL_NAME}_experiment")

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            lc_model=AGENT_MODEL_PATH,
            name="agent",
            code_paths=CODE_PATHS,
            pip_requirements=PIP_REQUIREMENTS,
            input_example=INPUT_EXAMPLE,
        )

    catalog = os.environ["UC_CATALOG"]
    schema = os.environ["UC_SCHEMA"]
    uc_name = f"{catalog}.{schema}.{MODEL_NAME}"

    registered = mlflow.register_model(model_info.model_uri, uc_name)
    print(f"Registered model: {uc_name}, version {registered.version}")

    return uc_name, registered.version


def create_or_update_endpoint(uc_name: str, version: str) -> str:
    settings = get_settings()
    w = WorkspaceClient()

    served_entity = ServedEntityInput(
        entity_name=uc_name,
        entity_version=version,
        workload_size="Small",
        scale_to_zero_enabled=True,
        environment_vars={
            # Secrets — never plaintext
            "DATABRICKS_HOST": f"{{{{secrets/{SECRET_SCOPE}/DATABRICKS_HOST}}}}",
            "DATABRICKS_TOKEN": f"{{{{secrets/{SECRET_SCOPE}/DATABRICKS_TOKEN}}}}",
            "DATABRICKS_MODEL": f"{{{{secrets/{SECRET_SCOPE}/DATABRICKS_MODEL}}}}",
            # Not secrets — the retriever needs these to reach the Vector Search index
            "VECTOR_SEARCH_ENDPOINT": settings["vs_endpoint"],
            "VECTOR_SEARCH_INDEX": settings["vs_index"],
            "EMBEDDINGS_ENDPOINT": settings["embeddings"],
        },
    )

    existing = [e.name for e in w.serving_endpoints.list()]

    if ENDPOINT_NAME not in existing:
      print(f"Creating endpoint '{ENDPOINT_NAME}'...")
      w.serving_endpoints.create(
          name=ENDPOINT_NAME,
          config=EndpointCoreConfigInput(
              name=ENDPOINT_NAME,
              served_entities=[served_entity],
          ),
      )
    else:
        print(f"Updating endpoint '{ENDPOINT_NAME}' to version {version}...")
        w.serving_endpoints.update_config(
            name=ENDPOINT_NAME,
            served_entities=[served_entity],
        )

    _wait_for_ready(w, ENDPOINT_NAME)

    endpoint_url = f"{settings['host']}/serving-endpoints/{ENDPOINT_NAME}/invocations"
    print(f"Endpoint URL: {endpoint_url}")
    return endpoint_url


def _wait_for_ready(w: WorkspaceClient, endpoint_name: str,
                     timeout_s: int = 1200, poll_s: int = 15) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        ep = w.serving_endpoints.get(endpoint_name)
        state = ep.state.ready if ep.state else None
        print(f"  endpoint state: {state}")
        if state is not None and str(state).endswith("READY"):
            print(f"Endpoint '{endpoint_name}' is READY.")
            return
        time.sleep(poll_s)
    raise TimeoutError(f"Endpoint '{endpoint_name}' did not reach READY in time.")


if __name__ == "__main__":
    name, ver = log_and_register()
    create_or_update_endpoint(name, ver)
