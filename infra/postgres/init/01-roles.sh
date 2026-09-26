#!/usr/bin/env bash
# Runs once, on first init of an empty postgres_data volume.
# The image has already created $POSTGRES_DB (synthetic_data) before this runs.
#
#   synth_owner - owns the schemas; used by Alembic migrations and LangGraph setup
#   synth_app   - runtime role for the API; data read/write only, no DDL
set -euo pipefail

: "${APP_DB_OWNER_PASSWORD:?APP_DB_OWNER_PASSWORD must be set}"
: "${APP_DB_PASSWORD:?APP_DB_PASSWORD must be set}"

# 1) roles + databases (connected to the maintenance DB)
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d postgres \
     -v owner_pw="$APP_DB_OWNER_PASSWORD" -v app_pw="$APP_DB_PASSWORD" <<'SQL'
CREATE ROLE synth_owner LOGIN PASSWORD :'owner_pw';
CREATE ROLE synth_app   LOGIN PASSWORD :'app_pw';
ALTER DATABASE synthetic_data OWNER TO synth_owner;
CREATE DATABASE synthetic_data_test OWNER synth_owner;
SQL

# 2) same schemas + privileges in both databases (:DBNAME is set by psql)
for db in synthetic_data synthetic_data_test; do
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$db" <<'SQL'
REVOKE ALL ON DATABASE :"DBNAME" FROM PUBLIC;
GRANT CONNECT, TEMP ON DATABASE :"DBNAME" TO synth_app;

ALTER SCHEMA public OWNER TO synth_owner;
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO synth_app;

CREATE SCHEMA IF NOT EXISTS langgraph AUTHORIZATION synth_owner;
GRANT USAGE ON SCHEMA langgraph TO synth_app;

-- anything synth_owner creates later (Alembic, LangGraph setup) is auto-granted to the app
ALTER DEFAULT PRIVILEGES FOR ROLE synth_owner IN SCHEMA public, langgraph
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO synth_app;
ALTER DEFAULT PRIVILEGES FOR ROLE synth_owner IN SCHEMA public, langgraph
  GRANT USAGE, SELECT ON SEQUENCES TO synth_app;
SQL
done
