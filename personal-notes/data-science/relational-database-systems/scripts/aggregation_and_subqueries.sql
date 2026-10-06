/*
Aggregation and Subqueries

Demonstrates aggregation and query composition using the Northwind
database. The examples cover aggregate functions, grouping, group
filtering, scalar and set subqueries, correlated subqueries, EXISTS,
and common table expressions.

The script emphasizes the distinction between filtering rows before
aggregation and filtering groups after aggregation, as well as the
different ways in which one query can provide input to another.
*/

-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Aggregate Functions
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Summarize the complete products relation with aggregate functions.
SELECT
    COUNT(*) AS product_count,
    MIN(price) AS minimum_price,
    MAX(price) AS maximum_price,
    AVG(price) AS average_price
FROM products;


-- COUNT(column) ignores NULL values while COUNT(*) counts rows.
SELECT
    COUNT(*) AS customer_count,
    COUNT(postalcode) AS customers_with_postalcode
FROM customers;


-- Calculate the total quantity represented by all order lines.
SELECT
    SUM(quantity) AS total_quantity
FROM orderdetails;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Grouping
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Form one group per country and count the customers in each group.
SELECT
    country,
    COUNT(*) AS customer_count
FROM customers
GROUP BY country
ORDER BY customer_count DESC, country;


-- Summarize products within each category.
SELECT
    categoryid,
    COUNT(*) AS product_count,
    AVG(price) AS average_price
FROM products
GROUP BY categoryid
ORDER BY categoryid;


-- Join before grouping to attach category names to the summary.
SELECT
    c.categoryid,
    c.categoryname,
    COUNT(*) AS product_count,
    AVG(p.price) AS average_price
FROM categories AS c
JOIN products AS p
    ON c.categoryid = p.categoryid
GROUP BY
    c.categoryid,
    c.categoryname
ORDER BY c.categoryid;


-- Calculate order values from quantities and current product prices.
SELECT
    od.orderid,
    SUM(od.quantity * p.price) AS order_value
FROM orderdetails AS od
JOIN products AS p
    ON od.productid = p.productid
GROUP BY od.orderid
ORDER BY order_value DESC
LIMIT 10;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- WHERE and HAVING
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- WHERE filters rows before they are assigned to groups.
SELECT
    country,
    COUNT(*) AS customer_count
FROM customers
WHERE country <> 'USA'
GROUP BY country
ORDER BY customer_count DESC, country;


-- HAVING filters groups after aggregation has taken place.
SELECT
    country,
    COUNT(*) AS customer_count
FROM customers
GROUP BY country
HAVING COUNT(*) >= 5
ORDER BY customer_count DESC, country;


-- Apply row-level and group-level filters in the same query.
SELECT
    categoryid,
    COUNT(*) AS product_count,
    AVG(price) AS average_price
FROM products
WHERE price >= 10
GROUP BY categoryid
HAVING COUNT(*) >= 5
ORDER BY categoryid;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Scalar Subqueries
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Compare each product price with a single value from a subquery.
SELECT
    productid,
    productname,
    price
FROM products
WHERE price > (
    SELECT AVG(price)
    FROM products
)
ORDER BY price DESC;


-- Use a scalar subquery as an expression in the result.
SELECT
    productid,
    productname,
    price,
    (
        SELECT AVG(price)
        FROM products
    ) AS overall_average_price
FROM products
ORDER BY productid
LIMIT 10;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Set Subqueries
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- IN tests membership in the set returned by the subquery.
SELECT
    customerid,
    customername,
    country
FROM customers
WHERE customerid IN (
    SELECT customerid
    FROM orders
    WHERE employeeid = 1
)
ORDER BY customerid;


-- Find products belonging to categories selected by another query.
SELECT
    productid,
    productname,
    categoryid
FROM products
WHERE categoryid IN (
    SELECT categoryid
    FROM categories
    WHERE categoryname IN ('Beverages', 'Seafood')
)
ORDER BY categoryid, productid;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- EXISTS
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- EXISTS tests whether the correlated subquery returns any row.
SELECT
    c.customerid,
    c.customername
FROM customers AS c
WHERE EXISTS (
    SELECT 1
    FROM orders AS o
    WHERE o.customerid = c.customerid
)
ORDER BY c.customerid;


-- NOT EXISTS expresses the absence of a related row directly.
SELECT
    c.customerid,
    c.customername
FROM customers AS c
WHERE NOT EXISTS (
    SELECT 1
    FROM orders AS o
    WHERE o.customerid = c.customerid
)
ORDER BY c.customerid;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Correlated Subqueries
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Compare a product with the average price of its own category.
SELECT
    p.productid,
    p.productname,
    p.categoryid,
    p.price
FROM products AS p
WHERE p.price > (
    SELECT AVG(p2.price)
    FROM products AS p2
    WHERE p2.categoryid = p.categoryid
)
ORDER BY p.categoryid, p.price DESC;


-- Count orders for each customer with a correlated scalar subquery.
SELECT
    c.customerid,
    c.customername,
    (
        SELECT COUNT(*)
        FROM orders AS o
        WHERE o.customerid = c.customerid
    ) AS order_count
FROM customers AS c
ORDER BY order_count DESC, c.customerid;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Common Table Expressions
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Aggregate order lines before filtering the resulting order values.
WITH order_values AS (
    SELECT
        od.orderid,
        SUM(od.quantity * p.price) AS order_value
    FROM orderdetails AS od
    JOIN products AS p
        ON od.productid = p.productid
    GROUP BY od.orderid
)
SELECT
    orderid,
    order_value
FROM order_values
WHERE order_value > 1000
ORDER BY order_value DESC;


-- Compose aggregation in stages using a common table expression.
WITH customer_orders AS (
    SELECT
        o.customerid,
        COUNT(*) AS order_count
    FROM orders AS o
    GROUP BY o.customerid
)
SELECT
    c.customerid,
    c.customername,
    co.order_count
FROM customer_orders AS co
JOIN customers AS c
    ON co.customerid = c.customerid
ORDER BY co.order_count DESC, c.customerid;


-- Aggregate order lines first, then summarize orders by customer.
WITH order_values AS (
    SELECT
        od.orderid,
        SUM(od.quantity * p.price) AS order_value
    FROM orderdetails AS od
    JOIN products AS p
        ON od.productid = p.productid
    GROUP BY od.orderid
)
SELECT
    o.customerid,
    c.customername,
    COUNT(*) AS order_count,
    SUM(ov.order_value) AS total_order_value,
    AVG(ov.order_value) AS average_order_value
FROM orders AS o
JOIN order_values AS ov
    ON o.orderid = ov.orderid
JOIN customers AS c
    ON o.customerid = c.customerid
GROUP BY
    o.customerid,
    c.customername
ORDER BY total_order_value DESC;