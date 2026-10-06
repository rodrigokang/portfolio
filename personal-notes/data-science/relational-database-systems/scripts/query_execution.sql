/*
Query Execution

Demonstrates physical query execution in PostgreSQL using EXPLAIN and
EXPLAIN ANALYZE on the Northwind database. The examples examine scans,
filters, joins, aggregation, sorting, and index-based access.

The focus is on reading physical execution plans and relating their
operators to the logical operations expressed by SQL.
*/

BEGIN;

-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Sequential Scan
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- EXPLAIN shows the selected plan without executing the query.
EXPLAIN
SELECT
    customerid,
    customername,
    country
FROM customers;


-- EXPLAIN ANALYZE executes the query and records observed statistics.
EXPLAIN ANALYZE
SELECT
    customerid,
    customername,
    country
FROM customers;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Filtering
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- A predicate can be evaluated as a filter during a sequential scan.
EXPLAIN ANALYZE
SELECT
    customerid,
    customername,
    country
FROM customers
WHERE country = 'Germany';


-- Include additional execution information in the plan.
EXPLAIN (ANALYZE, BUFFERS)
SELECT
    customerid,
    customername,
    country
FROM customers
WHERE country = 'Germany';


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Join Execution
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Observe the physical operators used to execute an equijoin.
EXPLAIN ANALYZE
SELECT
    o.orderid,
    o.orderdate,
    c.customername
FROM orders AS o
JOIN customers AS c
    ON o.customerid = c.customerid;


-- Add another relation to produce a larger join tree.
EXPLAIN ANALYZE
SELECT
    o.orderid,
    o.orderdate,
    c.customername,
    s.shippername
FROM orders AS o
JOIN customers AS c
    ON o.customerid = c.customerid
JOIN shippers AS s
    ON o.shipperid = s.shipperid;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Aggregation
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Observe how PostgreSQL implements grouping and aggregation.
EXPLAIN ANALYZE
SELECT
    customerid,
    COUNT(*) AS order_count
FROM orders
GROUP BY customerid;


-- Combine a join with grouping and aggregation.
EXPLAIN ANALYZE
SELECT
    c.customerid,
    c.customername,
    COUNT(o.orderid) AS order_count
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customerid = o.customerid
GROUP BY
    c.customerid,
    c.customername;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Sorting
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- ORDER BY may introduce an explicit physical Sort operator.
EXPLAIN ANALYZE
SELECT
    productid,
    productname,
    price
FROM products
ORDER BY price DESC;


-- Inspect additional information about the sorting operation.
EXPLAIN (ANALYZE, BUFFERS)
SELECT
    productid,
    productname,
    price
FROM products
ORDER BY price DESC;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Index Scan
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Create an index for a selective lookup on the orders relation.
CREATE INDEX idx_orders_orderdate
ON orders (orderdate);


-- Update planner statistics after creating the demonstration index.
ANALYZE orders;


-- Observe whether PostgreSQL chooses the available index.
EXPLAIN ANALYZE
SELECT
    orderid,
    customerid,
    orderdate
FROM orders
WHERE orderdate = DATE '1998-05-06';


-- A broader predicate may make a sequential scan more attractive.
EXPLAIN ANALYZE
SELECT
    orderid,
    customerid,
    orderdate
FROM orders
WHERE orderdate >= DATE '1996-01-01';


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Primary-Key Lookup
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- A primary key already has a supporting unique index.
EXPLAIN ANALYZE
SELECT
    orderid,
    customerid,
    orderdate
FROM orders
WHERE orderid = 10248;


-- Compare the plan with the index metadata.
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
  AND tablename = 'orders'
ORDER BY indexname;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Estimated and Actual Rows
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Compare planner estimates with the rows observed during execution.
EXPLAIN ANALYZE
SELECT
    orderid,
    customerid,
    orderdate
FROM orders
WHERE customerid = 1;


-- The plan reports estimated and observed cardinalities separately.
EXPLAIN ANALYZE
SELECT
    customerid,
    COUNT(*) AS order_count
FROM orders
GROUP BY customerid;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Execution Tree
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Read the plan from its child nodes toward the root operation.
EXPLAIN (ANALYZE, BUFFERS)
SELECT
    c.customername,
    COUNT(od.productid) AS products_ordered
FROM customers AS c
JOIN orders AS o
    ON c.customerid = o.customerid
JOIN orderdetails AS od
    ON o.orderid = od.orderid
GROUP BY
    c.customerid,
    c.customername
ORDER BY products_ordered DESC
LIMIT 10;


-- Discard the demonstration index.
ROLLBACK;