/*
Relational Queries

Demonstrates fundamental relational operations using the Northwind
database. The examples cover selection, projection, renaming, joins,
outer joins, set operations, and explicit ordering.

The script also illustrates the distinction between SQL bag semantics
and relational set semantics, and the effect of placing predicates in
ON or WHERE when using outer joins.
*/

-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Relational Queries
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Inspect a relation without assuming any inherent tuple ordering.
SELECT *
FROM customers
LIMIT 10;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Selection
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Select customers located in the United Kingdom.
SELECT
    customerid,
    customername,
    country
FROM customers
WHERE country = 'UK';


-- Combine predicates to define a more restrictive selection.
SELECT
    productid,
    productname,
    unit,
    price
FROM products
WHERE price >= 20
  AND price <= 50
ORDER BY price;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Projection
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- SQL projection preserves duplicates unless DISTINCT is requested.
SELECT country
FROM customers
ORDER BY country;


-- Remove duplicates to recover set-like projection semantics.
SELECT DISTINCT country
FROM customers
ORDER BY country;


-- Combine projection and selection in one query.
SELECT
    productid,
    productname,
    price
FROM products
WHERE price > 50
ORDER BY price DESC;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Renaming
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Rename relations and attributes within the scope of a query.
SELECT
    c.customerid AS customer_id,
    c.customername AS customer_name,
    c.country
FROM customers AS c
WHERE c.country = 'Germany'
ORDER BY c.customername;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Inner Joins
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Join customers with their associated orders.
SELECT
    o.orderid,
    o.orderdate,
    c.customerid,
    c.customername
FROM orders AS o
JOIN customers AS c
    ON o.customerid = c.customerid
ORDER BY o.orderid
LIMIT 20;


-- Follow the order-to-product relationship through orderdetails.
SELECT
    o.orderid,
    o.orderdate,
    p.productid,
    p.productname,
    od.quantity
FROM orders AS o
JOIN orderdetails AS od
    ON o.orderid = od.orderid
JOIN products AS p
    ON od.productid = p.productid
ORDER BY o.orderid, p.productid
LIMIT 20;


-- Join the principal relations associated with each order.
SELECT
    o.orderid,
    c.customername,
    e.firstname,
    e.lastname,
    s.shippername
FROM orders AS o
JOIN customers AS c
    ON o.customerid = c.customerid
JOIN employees AS e
    ON o.employeeid = e.employeeid
JOIN shippers AS s
    ON o.shipperid = s.shipperid
ORDER BY o.orderid
LIMIT 20;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Outer Joins
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Preserve every customer even when no matching order exists.
SELECT
    c.customerid,
    c.customername,
    o.orderid,
    o.orderdate
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customerid = o.customerid
ORDER BY c.customerid, o.orderid;


-- Use NULL on the joined relation to identify unmatched customers.
SELECT
    c.customerid,
    c.customername
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customerid = o.customerid
WHERE o.orderid IS NULL
ORDER BY c.customerid;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- ON and WHERE with Outer Joins
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Filter matches in ON while preserving every customer.
SELECT
    c.customerid,
    c.customername,
    o.orderid,
    o.orderdate
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customerid = o.customerid
   AND o.orderdate >= DATE '1997-01-01'
ORDER BY c.customerid, o.orderid;


-- Filter after the join, removing rows that fail the predicate.
SELECT
    c.customerid,
    c.customername,
    o.orderid,
    o.orderdate
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customerid = o.customerid
WHERE o.orderdate >= DATE '1997-01-01'
ORDER BY c.customerid, o.orderid;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Set Operations
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- UNION combines compatible results and removes duplicates.
SELECT country
FROM customers
WHERE country IN ('UK', 'Germany')

UNION

SELECT country
FROM suppliers
WHERE country IN ('UK', 'Germany')
ORDER BY country;


-- UNION ALL preserves duplicates when combining compatible results.
SELECT country
FROM customers
WHERE country IN ('UK', 'Germany')

UNION ALL

SELECT country
FROM suppliers
WHERE country IN ('UK', 'Germany')
ORDER BY country;


-- INTERSECT returns countries represented in both relations.
SELECT country
FROM customers

INTERSECT

SELECT country
FROM suppliers
ORDER BY country;


-- EXCEPT returns customer countries absent from suppliers.
SELECT country
FROM customers

EXCEPT

SELECT country
FROM suppliers
ORDER BY country;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Ordering
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Impose an explicit order on the query result.
SELECT
    productid,
    productname,
    price
FROM products
ORDER BY price DESC, productid
LIMIT 10;