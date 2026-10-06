/*
Integrity Constraints

Demonstrates relational integrity constraints in PostgreSQL using
temporary tables independent of the Northwind schema. The examples
cover primary keys, foreign keys, UNIQUE, NOT NULL, CHECK, composite
keys, and referential actions.

Some statements intentionally violate constraints. They are executed
inside savepoints so that PostgreSQL can report the violation while
allowing the script to continue with the remaining examples.
*/

-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Test Schema
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

BEGIN;

-- Temporary tables isolate the experiments from the Northwind schema.
CREATE TEMPORARY TABLE demo_customers (
    customer_id INTEGER PRIMARY KEY,
    email VARCHAR(100) NOT NULL UNIQUE,
    customer_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL
        CHECK (status IN ('active', 'inactive'))
);

CREATE TEMPORARY TABLE demo_products (
    product_id INTEGER PRIMARY KEY,
    product_name VARCHAR(100) NOT NULL,
    price NUMERIC(10, 2) NOT NULL
        CHECK (price > 0)
);

CREATE TEMPORARY TABLE demo_orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date DATE NOT NULL,
    FOREIGN KEY (customer_id)
        REFERENCES demo_customers (customer_id)
);

CREATE TEMPORARY TABLE demo_orderdetails (
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL
        CHECK (quantity > 0),
    PRIMARY KEY (order_id, product_id),
    FOREIGN KEY (order_id)
        REFERENCES demo_orders (order_id),
    FOREIGN KEY (product_id)
        REFERENCES demo_products (product_id)
);


-- Insert rows that satisfy all declared constraints.
INSERT INTO demo_customers (
    customer_id,
    email,
    customer_name,
    status
)
VALUES
    (1, 'alice@example.com', 'Alice', 'active'),
    (2, 'bob@example.com', 'Bob', 'inactive');

INSERT INTO demo_products (
    product_id,
    product_name,
    price
)
VALUES
    (101, 'Product A', 25.00),
    (102, 'Product B', 40.00);

INSERT INTO demo_orders (
    order_id,
    customer_id,
    order_date
)
VALUES
    (1001, 1, DATE '2026-01-15');

INSERT INTO demo_orderdetails (
    order_id,
    product_id,
    quantity
)
VALUES
    (1001, 101, 2),
    (1001, 102, 1);


-- Inspect the valid initial state.
SELECT *
FROM demo_customers
ORDER BY customer_id;

SELECT *
FROM demo_orders
ORDER BY order_id;

SELECT *
FROM demo_orderdetails
ORDER BY order_id, product_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Primary Key
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

SAVEPOINT primary_key_test;

-- Attempt to insert a duplicate primary key.
INSERT INTO demo_customers (
    customer_id,
    email,
    customer_name,
    status
)
VALUES
    (1, 'carol@example.com', 'Carol', 'active');

ROLLBACK TO SAVEPOINT primary_key_test;


-- The failed statement did not modify the relation.
SELECT *
FROM demo_customers
ORDER BY customer_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- NOT NULL
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

SAVEPOINT not_null_test;

-- Attempt to omit a value required by a NOT NULL constraint.
INSERT INTO demo_customers (
    customer_id,
    email,
    customer_name,
    status
)
VALUES
    (3, 'carol@example.com', NULL, 'active');

ROLLBACK TO SAVEPOINT not_null_test;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- UNIQUE
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

SAVEPOINT unique_test;

-- Attempt to reuse an email protected by a UNIQUE constraint.
INSERT INTO demo_customers (
    customer_id,
    email,
    customer_name,
    status
)
VALUES
    (3, 'alice@example.com', 'Carol', 'active');

ROLLBACK TO SAVEPOINT unique_test;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- CHECK
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

SAVEPOINT check_test;

-- Attempt to insert a value outside the permitted status domain.
INSERT INTO demo_customers (
    customer_id,
    email,
    customer_name,
    status
)
VALUES
    (3, 'carol@example.com', 'Carol', 'pending');

ROLLBACK TO SAVEPOINT check_test;


SAVEPOINT price_check_test;

-- Attempt to insert a product with an invalid price.
INSERT INTO demo_products (
    product_id,
    product_name,
    price
)
VALUES
    (103, 'Product C', -10.00);

ROLLBACK TO SAVEPOINT price_check_test;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Foreign Key
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

SAVEPOINT foreign_key_test;

-- Attempt to reference a customer that does not exist.
INSERT INTO demo_orders (
    order_id,
    customer_id,
    order_date
)
VALUES
    (1002, 999, DATE '2026-01-16');

ROLLBACK TO SAVEPOINT foreign_key_test;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Composite Primary Key
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

SAVEPOINT composite_key_test;

-- The pair already identifies an existing order line.
INSERT INTO demo_orderdetails (
    order_id,
    product_id,
    quantity
)
VALUES
    (1001, 101, 5);

ROLLBACK TO SAVEPOINT composite_key_test;


-- A different product produces a different composite key.
INSERT INTO demo_products (
    product_id,
    product_name,
    price
)
VALUES
    (103, 'Product C', 15.00);

INSERT INTO demo_orderdetails (
    order_id,
    product_id,
    quantity
)
VALUES
    (1001, 103, 5);


SELECT *
FROM demo_orderdetails
ORDER BY order_id, product_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Referential Actions
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

SAVEPOINT restricted_delete_test;

-- The default action prevents deletion of a referenced customer.
DELETE FROM demo_customers
WHERE customer_id = 1;

ROLLBACK TO SAVEPOINT restricted_delete_test;


-- Create a separate example to demonstrate cascading deletion.
CREATE TEMPORARY TABLE demo_departments (
    department_id INTEGER PRIMARY KEY,
    department_name VARCHAR(100) NOT NULL
);

CREATE TEMPORARY TABLE demo_employees (
    employee_id INTEGER PRIMARY KEY,
    department_id INTEGER NOT NULL,
    employee_name VARCHAR(100) NOT NULL,
    FOREIGN KEY (department_id)
        REFERENCES demo_departments (department_id)
        ON DELETE CASCADE
);


INSERT INTO demo_departments (
    department_id,
    department_name
)
VALUES
    (1, 'Sales'),
    (2, 'Operations');

INSERT INTO demo_employees (
    employee_id,
    department_id,
    employee_name
)
VALUES
    (1, 1, 'Employee A'),
    (2, 1, 'Employee B'),
    (3, 2, 'Employee C');


-- Delete a parent row whose foreign key uses ON DELETE CASCADE.
DELETE FROM demo_departments
WHERE department_id = 1;


-- Employees belonging to the deleted department are removed as well.
SELECT *
FROM demo_departments
ORDER BY department_id;

SELECT *
FROM demo_employees
ORDER BY employee_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Final State
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Valid operations remain while rejected operations leave no changes.
SELECT *
FROM demo_customers
ORDER BY customer_id;

SELECT *
FROM demo_products
ORDER BY product_id;

SELECT *
FROM demo_orders
ORDER BY order_id;

SELECT *
FROM demo_orderdetails
ORDER BY order_id, product_id;


-- Roll back the complete experiment.
ROLLBACK;