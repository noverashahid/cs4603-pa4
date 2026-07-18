"""Corpus ingestion into Databricks Vector Search (Task 0.3 / rag/ingest.py).

Run inside a Databricks notebook (needs Spark + ai_parse_document/ai_prep_search).
Mirror PA2 Part 1:

TODO:
  - `build_chunks_table(spark, volume_path, chunks_table)`: parse the PDF with
    ai_parse_document, chunk with ai_prep_search into a Delta table with columns
    chunk_id, chunk_to_retrieve, chunk_to_embed, source, page. Enable Change Data
    Feed on the table.
  - `create_index()`: create a STANDARD Vector Search endpoint and a TRIGGERED
    Delta Sync index (primary_key='chunk_id',
    embedding_source_column='chunk_to_retrieve',
    embedding_model_endpoint_name=$EMBEDDINGS_ENDPOINT).
"""

from __future__ import annotations
import os
import time

from databricks.vector_search.client import VectorSearchClient

def build_chunks_table(spark, volume_path: str, chunks_table: str) -> None:
    # Parse and chunk the raw PDF bytes using the same variant schema returned
    # by ai_prep_search in Databricks SQL.
    spark.sql(f"""
        CREATE OR REPLACE TEMPORARY VIEW _staging_pa4_chunks AS
        SELECT
            path AS source,
            ai_prep_search(ai_parse_document(content)) AS search_chunks
        FROM READ_FILES('{volume_path}', format => 'binaryFile')
    """)

    # Select the columns the Delta Sync index expects. ai_prep_search returns
    # variant chunks with id/content/page_number fields, not Python-style
    # chunk_id/chunk_to_retrieve attributes.
    spark.sql(f"""
        CREATE OR REPLACE TABLE {chunks_table} AS
        SELECT
            md5(concat(source, CAST(chunk:id::string AS STRING))) AS chunk_id,
            chunk:content::string AS chunk_to_retrieve,
            chunk:content::string AS chunk_to_embed,
            source,
            chunk:page_number::int AS page
        FROM _staging_pa4_chunks,
        LATERAL variant_explode(search_chunks) AS t(pos, key, chunk)
    """)

    # Delta Sync requires Change Data Feed enabled on the source table
    spark.sql(f"""
        ALTER TABLE {chunks_table}
        SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)

    row_count = spark.table(chunks_table).count()
    print(f"Chunked table ready: {chunks_table} ({row_count} rows)")


def create_index() -> None:
    catalog = os.environ["UC_CATALOG"]
    schema = os.environ["UC_SCHEMA"]
    endpoint_name = os.environ["VECTOR_SEARCH_ENDPOINT"]
    index_name = os.environ["VECTOR_SEARCH_INDEX"]
    embeddings_endpoint = os.environ["EMBEDDINGS_ENDPOINT"]
    chunks_table = (
        os.environ.get("PA4_CHUNKS_TABLE")
        or os.environ.get("SOURCE_TABLE")
        or f"{catalog}.{schema}.pa4_chunks"
    )

    vsc = VectorSearchClient(disable_notice=True)

    # endpoint
    existing_endpoints = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
    if endpoint_name not in existing_endpoints:
        print(f"Creating endpoint '{endpoint_name}' (STANDARD)...")
        vsc.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
        _wait_for_endpoint_online(vsc, endpoint_name)
    else:
        print(f"Endpoint '{endpoint_name}' already exists.")

    # index
    existing_indexes = [
        idx["name"] for idx in vsc.list_indexes(endpoint_name).get("vector_indexes", [])
    ]
    if index_name not in existing_indexes:
        print(f"Creating Delta Sync index '{index_name}'...")
        vsc.create_delta_sync_index(
            endpoint_name=endpoint_name,
            index_name=index_name,
            source_table_name=chunks_table,
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_source_column="chunk_to_retrieve",
            embedding_model_endpoint_name=embeddings_endpoint,
        )
    else:
        print(f"Index '{index_name}' already exists — triggering sync.")
        vsc.get_index(endpoint_name, index_name).sync()

    _wait_for_index_ready(vsc, endpoint_name, index_name)


def _wait_for_endpoint_online(vsc: VectorSearchClient, endpoint_name: str,
                               timeout_s: int = 1200, poll_s: int = 15) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        status = vsc.get_endpoint(endpoint_name)
        state = status.get("endpoint_status", {}).get("state", "UNKNOWN")
        print(f"  endpoint state: {state}")
        if state == "ONLINE":
            return
        time.sleep(poll_s)
    raise TimeoutError(f"Endpoint '{endpoint_name}' did not come ONLINE in time.")


def _wait_for_index_ready(vsc: VectorSearchClient, endpoint_name: str, index_name: str,
                           timeout_s: int = 1800, poll_s: int = 20) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        idx = vsc.get_index(endpoint_name, index_name)
        status = idx.describe().get("status", {})
        print(
            "  index ready: "
            f"{status.get('ready')}, detail: {status.get('detailed_state')}, "
            f"indexed rows: {status.get('indexed_row_count')}"
        )
        if status.get("ready") is True:
            print(f"Index '{index_name}' is READY.")
            return
        time.sleep(poll_s)
    raise TimeoutError(f"Index '{index_name}' did not become READY in time.")
