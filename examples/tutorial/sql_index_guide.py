#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://gpt-index.readthedocs.io/en/latest/examples/index_structs/struct_indices/SQLIndexDemo.html
"""
import argparse

from langchain.llms.openai import OpenAI
from llama_index import SQLDatabase, ServiceContext
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from llama_index.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index import VectorStoreIndex
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
)
from sqlalchemy import insert
from sqlalchemy import text

from project_settings import project_path
import project_settings as settings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="What is instructGPT ?", type=str)

    parser.add_argument(
        "--input_dir",
        default=(project_path / "cache/arxiv_pdf").as_posix(),
        type=str
    )
    parser.add_argument("--persist_dir", default=None, type=str)

    parser.add_argument(
        "--openai_api_key",
        default=settings.environment.get("openai_api_key", default=None, dtype=str),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    engine = create_engine("sqlite:///:memory:")
    metadata_obj = MetaData()

    # create city SQL table
    table_name = "city_stats"
    city_stats_table = Table(
        table_name,
        metadata_obj,
        Column("city_name", String(16), primary_key=True),
        Column("population", Integer),
        Column("country", String(16), nullable=False),
    )
    metadata_obj.create_all(engine)
    rows = [
        {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
        {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
        {"city_name": "Chicago", "population": 2679000, "country": "United States"},
        {"city_name": "Seoul", "population": 9776000, "country": "South Korea"},
    ]
    for row in rows:
        stmt = insert(city_stats_table).values(**row)
        with engine.connect() as connection:
            cursor = connection.execute(stmt)
            connection.commit()

    # view current table
    stmt = select(city_stats_table.c["city_name", "population", "country"]).select_from(
        city_stats_table
    )
    with engine.connect() as connection:
        results = connection.execute(stmt).fetchall()
        print(results)

    with engine.connect() as con:
        rows = con.execute(text("SELECT city_name from city_stats"))
        for row in rows:
            print(row)

    # llama
    service_context = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                openai_api_key=args.openai_api_key
            )
        ),
    )
    sql_database = SQLDatabase(engine, include_tables=["city_stats"])
    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["city_stats"],
        service_context=service_context
    )
    query_str = "Which city has the highest population?"
    response = query_engine.query(query_str)
    print(response)
    return


if __name__ == '__main__':
    main()
