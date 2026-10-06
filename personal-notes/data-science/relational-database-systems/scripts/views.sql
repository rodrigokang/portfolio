/*
Views

Demonstrates regular and materialized views in PostgreSQL using the
Northwind database. The examples cover logical abstraction, filtering,
aggregation, view updates, materialization, and explicit refresh.

Changes to Northwind data are performed inside a transaction and are
rolled back at the end of the script. Views created for the examples
are also removed before the transaction is completed.
*/

BEGIN;

-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Simple Views
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Encapsulate a filtered projection of the customers relation.
CREATE VIEW uk_customers AS
SELECT
    customerid,
    customername,
    city,
    country
FROM customers
WHERE country = 'UK';


-- Query the view as a relation.
SELECT *
FROM uk_customers
ORDER BY customerid;


-- Additional predicates can be applied when querying the view.
SELECT
    customerid,
    customername,
    city
FROM uk_customers
WHERE city = 'London'
ORDER BY customerid;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Views over Joins
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Hide the joins required to reconstruct descriptive order data.
CREATE VIEW order_summary AS
SELECT
    o.orderid,
    o.orderdate,
    c.customerid,
    c.customername,
    e.employeeid,
    e.firstname,
    e.lastname,
    s.shipperid,
    s.shippername
FROM orders AS o
JOIN customers AS c
    ON o.customerid = c.customerid
JOIN employees AS e
    ON o.employeeid = e.employeeid
JOIN shippers AS s
    ON o.shipperid = s.shipperid;


-- Query the derived relation without repeating the underlying joins.
SELECT
    orderid,
    orderdate,
    customername,
    firstname,
    lastname,
    shippername
FROM order_summary
ORDER BY orderid
LIMIT 20;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Views over Aggregation
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Represent customer order counts as a reusable derived relation.
CREATE VIEW customer_order_counts AS
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


-- Inspect the aggregated relation.
SELECT *
FROM customer_order_counts
ORDER BY order_count DESC, customerid;


-- Apply a predicate to the result produced by the view.
SELECT
    customerid,
    customername,
    order_count
FROM customer_order_counts
WHERE order_count >= 10
ORDER BY order_count DESC, customerid;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Updatable Views
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- A simple view over one base relation can be automatically updatable.
CREATE VIEW german_customers AS
SELECT
    customerid,
    customername,
    address,
    city,
    postalcode,
    country
FROM customers
WHERE country = 'Germany'
WITH LOCAL CHECK OPTION;


-- Inspect one row before modifying it through the view.
SELECT
    customerid,
    customername,
    city
FROM german_customers
ORDER BY customerid
LIMIT 1;


SAVEPOINT view_update;


-- Update the underlying relation indirectly through the view.
UPDATE german_customers
SET city = 'Test City'
WHERE customerid = (
    SELECT customerid
    FROM german_customers
    ORDER BY customerid
    LIMIT 1
);


-- The modification is visible through the view.
SELECT
    customerid,
    customername,
    city
FROM german_customers
WHERE city = 'Test City';


-- The same modification exists in the underlying relation.
SELECT
    customerid,
    customername,
    city
FROM customers
WHERE city = 'Test City';


-- Restore the original Northwind data.
ROLLBACK TO SAVEPOINT view_update;


-- Verify that the temporary modification has been undone.
SELECT
    customerid,
    customername,
    city
FROM customers
WHERE city = 'Test City';


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Materialized Views
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Store an aggregated query result as a materialized view.
CREATE MATERIALIZED VIEW product_category_summary AS
SELECT
    c.categoryid,
    c.categoryname,
    COUNT(p.productid) AS product_count,
    AVG(p.price) AS average_price
FROM categories AS c
LEFT JOIN products AS p
    ON c.categoryid = p.categoryid
GROUP BY
    c.categoryid,
    c.categoryname;


-- Read the stored result.
SELECT *
FROM product_category_summary
ORDER BY categoryid;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Materialized View Refresh
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

SAVEPOINT materialized_view_test;


-- Inspect the stored summary before changing the base relation.
SELECT *
FROM product_category_summary
WHERE categoryid = (
    SELECT categoryid
    FROM products
    WHERE productid = 1
);


-- Change a base value used by the materialized view.
UPDATE products
SET price = price + 100
WHERE productid = 1;


-- Confirm that the underlying base relation has changed.
SELECT
    productid,
    productname,
    price
FROM products
WHERE productid = 1;


-- The materialized result does not change automatically.
SELECT *
FROM product_category_summary
WHERE categoryid = (
    SELECT categoryid
    FROM products
    WHERE productid = 1
);


-- Recompute the stored result from the current base relations.
REFRESH MATERIALIZED VIEW product_category_summary;


-- The refreshed result now reflects the modified base relation.
SELECT *
FROM product_category_summary
WHERE categoryid = (
    SELECT categoryid
    FROM products
    WHERE productid = 1
);


-- Restore both data and materialized state to the savepoint.
ROLLBACK TO SAVEPOINT materialized_view_test;


-- Confirm that the base value has returned to its original state.
SELECT
    productid,
    productname,
    price
FROM products
WHERE productid = 1;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Cleanup
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

DROP MATERIALIZED VIEW product_category_summary;
DROP VIEW german_customers;
DROP VIEW customer_order_counts;
DROP VIEW order_summary;
DROP VIEW uk_customers;


-- Discard the complete experiment.
ROLLBACK;