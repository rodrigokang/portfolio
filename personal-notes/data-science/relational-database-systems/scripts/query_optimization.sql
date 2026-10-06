/*
Query Optimization

Demonstrates query optimization in PostgreSQL using the Northwind
database. The examples examine statistics, cardinality estimates,
selectivity, indexes, composite search keys, and alternative plans.

The focus is on how PostgreSQL evaluates physical alternatives rather
than on treating any particular execution strategy as universally
preferable.
*/

BEGIN;

-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Planner Statistics
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Refresh statistics used by the optimizer for selected relations.
ANALYZE customers;
ANALYZE orders;
ANALYZE products;


-- Inspect approximate relation cardinalities recorded by PostgreSQL.
SELECT
    relname,
    reltuples::BIGINT AS estimated_rows
FROM pg_class
WHERE relname IN ('customers', 'orders', 'products')
ORDER BY relname;


-- Compare catalog estimates with exact row counts.
SELECT
    'customers' AS relation,
    COUNT(*) AS actual_rows
FROM customers

UNION ALL

SELECT
    'orders',
    COUNT(*)
FROM orders

UNION ALL

SELECT
    'products',
    COUNT(*)
FROM products;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Column Statistics
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Inspect selected statistics available to the query planner.
SELECT
    tablename,
    attname,
    null_frac,
    n_distinct,
    most_common_vals,
    most_common_freqs
FROM pg_stats
WHERE schemaname = 'public'
  AND tablename = 'customers'
  AND attname = 'country';


-- Statistics summarize data rather than storing exact query results.
SELECT
    country,
    COUNT(*) AS customer_count
FROM customers
GROUP BY country
ORDER BY customer_count DESC, country;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Cardinality Estimates
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Compare estimated and observed rows for a selection predicate.
EXPLAIN ANALYZE
SELECT
    customerid,
    customername,
    country
FROM customers
WHERE country = 'Germany';


-- Compare estimates for a different value of the same attribute.
EXPLAIN ANALYZE
SELECT
    customerid,
    customername,
    country
FROM customers
WHERE country = 'Argentina';


-- Estimates propagate through joins as well as base-relation scans.
EXPLAIN ANALYZE
SELECT
    o.orderid,
    o.orderdate,
    c.customername
FROM orders AS o
JOIN customers AS c
    ON o.customerid = c.customerid
WHERE c.country = 'Germany';


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Selectivity
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- A narrow predicate selects a small portion of the orders relation.
EXPLAIN ANALYZE
SELECT
    orderid,
    customerid,
    orderdate
FROM orders
WHERE orderdate = DATE '1998-05-06';


-- A broad predicate selects a much larger portion of the relation.
EXPLAIN ANALYZE
SELECT
    orderid,
    customerid,
    orderdate
FROM orders
WHERE orderdate >= DATE '1996-01-01';


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Optimization before an Index
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Record the selected plan before adding an explicit search structure.
EXPLAIN ANALYZE
SELECT
    orderid,
    customerid,
    orderdate
FROM orders
WHERE orderdate = DATE '1998-05-06';


-- Add an index that provides another physical access alternative.
CREATE INDEX idx_orders_orderdate_opt
ON orders (orderdate);

ANALYZE orders;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Optimization after an Index
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- The optimizer now has an additional candidate access path.
EXPLAIN ANALYZE
SELECT
    orderid,
    customerid,
    orderdate
FROM orders
WHERE orderdate = DATE '1998-05-06';


-- The same index need not be attractive for a broad predicate.
EXPLAIN ANALYZE
SELECT
    orderid,
    customerid,
    orderdate
FROM orders
WHERE orderdate >= DATE '1996-01-01';


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Composite Indexes
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Create a composite search key matching a common predicate pattern.
CREATE INDEX idx_orders_customer_date_opt
ON orders (customerid, orderdate);

ANALYZE orders;


-- Both attributes can participate in the index search condition.
EXPLAIN ANALYZE
SELECT
    orderid,
    customerid,
    orderdate
FROM orders
WHERE customerid = 20
  AND orderdate >= DATE '1997-01-01';


-- The leading attribute can be used independently.
EXPLAIN ANALYZE
SELECT
    orderid,
    customerid,
    orderdate
FROM orders
WHERE customerid = 20;


-- The second attribute alone presents a different access problem.
EXPLAIN ANALYZE
SELECT
    orderid,
    customerid,
    orderdate
FROM orders
WHERE orderdate >= DATE '1997-01-01';


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Ordering and Indexes
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- An ordered index may provide rows in a useful physical order.
CREATE INDEX idx_products_price_opt
ON products (price);

ANALYZE products;


-- Observe whether an explicit Sort operator is required.
EXPLAIN ANALYZE
SELECT
    productid,
    productname,
    price
FROM products
ORDER BY price;


-- LIMIT can change the relative attractiveness of access paths.
EXPLAIN ANALYZE
SELECT
    productid,
    productname,
    price
FROM products
ORDER BY price
LIMIT 5;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Join Optimization
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Observe the join order and physical join algorithms selected.
EXPLAIN ANALYZE
SELECT
    c.customername,
    o.orderid,
    od.productid,
    p.productname
FROM customers AS c
JOIN orders AS o
    ON c.customerid = o.customerid
JOIN orderdetails AS od
    ON o.orderid = od.orderid
JOIN products AS p
    ON od.productid = p.productid
WHERE c.country = 'Germany';


-- A selective predicate can influence the surrounding plan.
EXPLAIN ANALYZE
SELECT
    c.customername,
    o.orderid,
    od.productid,
    p.productname
FROM customers AS c
JOIN orders AS o
    ON c.customerid = o.customerid
JOIN orderdetails AS od
    ON o.orderid = od.orderid
JOIN products AS p
    ON od.productid = p.productid
WHERE c.customerid = 20;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Alternative Plans
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Save the planner setting before the controlled experiment.
SHOW enable_seqscan;


-- Discourage sequential scans to expose an alternative candidate plan.
SET LOCAL enable_seqscan = off;


EXPLAIN
SELECT
    orderid,
    customerid,
    orderdate
FROM orders
WHERE orderdate = DATE '1998-05-06';


-- Restore normal planner behaviour for subsequent statements.
SET LOCAL enable_seqscan = on;


EXPLAIN
SELECT
    orderid,
    customerid,
    orderdate
FROM orders
WHERE orderdate = DATE '1998-05-06';


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Estimated Cost and Actual Execution
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Cost estimates and measured execution statistics are distinct.
EXPLAIN (ANALYZE, BUFFERS)
SELECT
    c.customername,
    COUNT(o.orderid) AS order_count
FROM customers AS c
JOIN orders AS o
    ON c.customerid = o.customerid
GROUP BY
    c.customerid,
    c.customername
ORDER BY order_count DESC
LIMIT 10;


-- Discard all demonstration indexes and complete the experiment.
ROLLBACK;