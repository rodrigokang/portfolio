/*
Indexes

Demonstrates index definition and inspection in PostgreSQL using the
Northwind database. The examples cover indexes created by constraints,
single-column indexes, composite indexes, partial indexes, and unique
indexes.

The script focuses on index structures as database objects rather than
on optimizer decisions. Query plans and index selection are examined
separately in the query execution and optimization scripts.
*/

BEGIN;

-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Existing Indexes
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Inspect indexes already defined on the Northwind tables.
SELECT
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;


-- Primary keys normally have supporting unique indexes.
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
  AND tablename = 'customers'
ORDER BY indexname;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Single-Column Index
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Create a B-tree index on a column used for customer filtering.
CREATE INDEX idx_customers_country
ON customers (country);


-- Inspect the index definition stored in the PostgreSQL catalog.
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
  AND indexname = 'idx_customers_country';


-- The indexed attribute can be used as a search predicate.
SELECT
    customerid,
    customername,
    country
FROM customers
WHERE country = 'Germany'
ORDER BY customerid;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Composite Index
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Create an index whose search key contains two attributes.
CREATE INDEX idx_orders_customer_date
ON orders (customerid, orderdate);


-- Inspect the order of attributes in the composite search key.
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
  AND indexname = 'idx_orders_customer_date';


-- Use both indexed attributes in a query predicate.
SELECT
    orderid,
    customerid,
    orderdate
FROM orders
WHERE customerid = 1
  AND orderdate >= DATE '1997-01-01'
ORDER BY orderdate;


-- Use only the leading attribute of the composite index.
SELECT
    orderid,
    customerid,
    orderdate
FROM orders
WHERE customerid = 1
ORDER BY orderdate;


-- Use only the second attribute of the composite index.
SELECT
    orderid,
    customerid,
    orderdate
FROM orders
WHERE orderdate >= DATE '1997-01-01'
ORDER BY orderdate
LIMIT 20;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Partial Index
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Index only products satisfying a specified predicate.
CREATE INDEX idx_products_high_price
ON products (price)
WHERE price >= 50;


-- The predicate is part of the index definition.
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
  AND indexname = 'idx_products_high_price';


-- Query rows belonging to the subset represented by the index.
SELECT
    productid,
    productname,
    price
FROM products
WHERE price >= 50
ORDER BY price;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Unique Index
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Use a temporary relation to demonstrate uniqueness independently.
CREATE TEMPORARY TABLE demo_accounts (
    account_id INTEGER PRIMARY KEY,
    username VARCHAR(100) NOT NULL
);


-- Define uniqueness directly through an index.
CREATE UNIQUE INDEX idx_demo_accounts_username
ON demo_accounts (username);


INSERT INTO demo_accounts (
    account_id,
    username
)
VALUES
    (1, 'alice'),
    (2, 'bob');


-- Inspect the relation before testing the unique index.
SELECT *
FROM demo_accounts
ORDER BY account_id;


SAVEPOINT unique_index_test;


-- Attempt to insert a duplicate indexed value.
INSERT INTO demo_accounts (
    account_id,
    username
)
VALUES
    (3, 'alice');

ROLLBACK TO SAVEPOINT unique_index_test;


-- The rejected row did not modify the relation.
SELECT *
FROM demo_accounts
ORDER BY account_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Constraint Indexes
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Inspect indexes associated with the temporary demonstration table.
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname LIKE 'pg_temp%'
  AND tablename = 'demo_accounts'
ORDER BY indexname;


-- The primary key and explicit unique index are separate objects.
SELECT
    i.relname AS index_name,
    ix.indisprimary AS is_primary,
    ix.indisunique AS is_unique
FROM pg_class AS t
JOIN pg_index AS ix
    ON t.oid = ix.indrelid
JOIN pg_class AS i
    ON i.oid = ix.indexrelid
WHERE t.relname = 'demo_accounts'
ORDER BY i.relname;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Indexes and Queries
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- An available index does not imply that PostgreSQL will use it.
SELECT
    customerid,
    customername,
    country
FROM customers
WHERE country = 'Germany';


-- Index choice depends on the physical execution plan.
SELECT
    productid,
    productname,
    price
FROM products
WHERE price >= 50;


-- Plan selection is examined explicitly in a later script.


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Final Index State
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Inspect the demonstration indexes before discarding them.
SELECT
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
  AND indexname IN (
      'idx_customers_country',
      'idx_orders_customer_date',
      'idx_products_high_price'
  )
ORDER BY tablename, indexname;


-- Discard every index and temporary object created by the experiment.
ROLLBACK;