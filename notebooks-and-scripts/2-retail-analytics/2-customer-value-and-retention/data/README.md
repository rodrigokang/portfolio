# AdventureWorks SQLite Starter Kit

This folder contains a SQLite-oriented starter conversion for the AdventureWorks OLTP CSV files, focused on the **Customer Value and Retention** portfolio project.

## Included files

- `create_adventureworks_sqlite.sql`: SQLite schema for the selected Customer Analytics subset.
- `load_adventureworks_sqlite.py`: Python loader for the official CSV files.
- `tables_metadata.json`: table inventory, original CSV name, delimiter and row counts.
- `audit_summary.csv`: compact audit of the selected files.

## Scope

The starter subset includes 18 tables from Sales, Production, Person and Purchasing. SQL Server-only features from the original script—stored procedures, XML schema collections, full-text indexes, extended properties and most triggers—are intentionally excluded.

## Usage

```bash
python load_adventureworks_sqlite.py --csv-dir ./adventure-works --db adventureworks.db
```
