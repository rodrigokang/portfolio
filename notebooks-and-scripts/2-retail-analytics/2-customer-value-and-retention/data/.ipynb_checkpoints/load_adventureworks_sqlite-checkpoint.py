"""
Load a Customer Analytics subset of Microsoft AdventureWorks OLTP CSV files into SQLite.

Usage:
    python load_adventureworks_sqlite.py --csv-dir ./adventure-works --db adventureworks.db
"""
from __future__ import annotations
import argparse, csv, json, sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent

def read_rows(path: Path, delimiter: str):
    if delimiter == '+|':
        text = path.read_text(encoding='utf-8', errors='replace')
        for record in text.split('&|\n'):
            if record:
                yield record.split('+|')
    else:
        with path.open('r', encoding='utf-8', newline='', errors='replace') as f:
            yield from csv.reader(f, delimiter='\t')

def main(csv_dir: Path, db_path: Path):
    metadata = json.loads((HERE / 'tables_metadata.json').read_text(encoding='utf-8'))
    schema_sql = (HERE / 'create_adventureworks_sqlite.sql').read_text(encoding='utf-8')
    conn = sqlite3.connect(db_path)
    try:
        conn.execute('PRAGMA foreign_keys = OFF;')
        conn.executescript(schema_sql.replace('PRAGMA foreign_keys = ON;', 'PRAGMA foreign_keys = OFF;'))
        for info in metadata:
            table = info['table']
            csv_path = csv_dir / info['csv']
            columns = info['column_names']
            placeholders = ','.join(['?'] * len(columns))
            col_sql = ','.join([f'"{c}"' for c in columns])
            insert_sql = f'INSERT INTO "{table}" ({col_sql}) VALUES ({placeholders})'
            rows = []
            skipped = 0
            for row in read_rows(csv_path, info['delimiter']):
                if len(row) != len(columns):
                    skipped += 1
                    continue
                rows.append([None if value == '' else value for value in row])
            conn.executemany(insert_sql, rows)
            conn.commit()
            print(f'{table:35s} inserted={len(rows):7d} skipped={skipped}')
        conn.execute('PRAGMA foreign_keys = ON;')
        problems = conn.execute('PRAGMA foreign_key_check;').fetchall()
        if problems:
            print('\nForeign key check returned issues:')
            for p in problems[:20]:
                print(p)
            raise SystemExit(f'{len(problems)} foreign key issues found.')
        print(f'\nDone: {db_path}')
    finally:
        conn.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-dir', type=Path, required=True)
    parser.add_argument('--db', type=Path, default=Path('adventureworks.db'))
    args = parser.parse_args()
    main(args.csv_dir, args.db)
