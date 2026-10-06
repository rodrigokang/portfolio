/*
Recursive Queries

Demonstrates recursive SQL in PostgreSQL using a temporary employee
hierarchy. The examples cover recursive common table expressions,
hierarchical traversal, depth calculation, path construction, and
upward traversal through a self-referencing relation.

A temporary relation is used because the Northwind schema employed by
these notes does not contain a recursive employee relationship.
*/

BEGIN;

-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Recursive Relation
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Represent an organizational hierarchy with a self-referencing key.
CREATE TEMPORARY TABLE employee_hierarchy (
    employee_id INTEGER PRIMARY KEY,
    employee_name VARCHAR(100) NOT NULL,
    manager_id INTEGER,
    FOREIGN KEY (manager_id)
        REFERENCES employee_hierarchy (employee_id)
);


-- Insert the hierarchy from the root toward its descendants.
INSERT INTO employee_hierarchy (
    employee_id,
    employee_name,
    manager_id
)
VALUES
    (1, 'Alice', NULL),
    (2, 'Bob', 1),
    (3, 'Carol', 1),
    (4, 'David', 2),
    (5, 'Emma', 2),
    (6, 'Frank', 3),
    (7, 'Grace', 4),
    (8, 'Henry', 4);


-- Inspect the adjacency-list representation directly.
SELECT
    employee_id,
    employee_name,
    manager_id
FROM employee_hierarchy
ORDER BY employee_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Recursive Traversal
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Start at the root and repeatedly find direct descendants.
WITH RECURSIVE hierarchy AS (
    SELECT
        employee_id,
        employee_name,
        manager_id
    FROM employee_hierarchy
    WHERE manager_id IS NULL

    UNION ALL

    SELECT
        e.employee_id,
        e.employee_name,
        e.manager_id
    FROM employee_hierarchy AS e
    JOIN hierarchy AS h
        ON e.manager_id = h.employee_id
)
SELECT
    employee_id,
    employee_name,
    manager_id
FROM hierarchy
ORDER BY employee_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Hierarchy Depth
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Carry a depth value through successive recursive iterations.
WITH RECURSIVE hierarchy AS (
    SELECT
        employee_id,
        employee_name,
        manager_id,
        0 AS depth
    FROM employee_hierarchy
    WHERE manager_id IS NULL

    UNION ALL

    SELECT
        e.employee_id,
        e.employee_name,
        e.manager_id,
        h.depth + 1
    FROM employee_hierarchy AS e
    JOIN hierarchy AS h
        ON e.manager_id = h.employee_id
)
SELECT
    employee_id,
    employee_name,
    manager_id,
    depth
FROM hierarchy
ORDER BY depth, employee_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Path Construction
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Accumulate employee names to make each traversal path explicit.
WITH RECURSIVE hierarchy AS (
    SELECT
        employee_id,
        employee_name,
        manager_id,
        employee_name::TEXT AS path
    FROM employee_hierarchy
    WHERE manager_id IS NULL

    UNION ALL

    SELECT
        e.employee_id,
        e.employee_name,
        e.manager_id,
        h.path || ' -> ' || e.employee_name
    FROM employee_hierarchy AS e
    JOIN hierarchy AS h
        ON e.manager_id = h.employee_id
)
SELECT
    employee_id,
    employee_name,
    path
FROM hierarchy
ORDER BY employee_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Subtree Traversal
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Use Bob rather than the root as the non-recursive starting row.
WITH RECURSIVE subtree AS (
    SELECT
        employee_id,
        employee_name,
        manager_id,
        0 AS relative_depth
    FROM employee_hierarchy
    WHERE employee_name = 'Bob'

    UNION ALL

    SELECT
        e.employee_id,
        e.employee_name,
        e.manager_id,
        s.relative_depth + 1
    FROM employee_hierarchy AS e
    JOIN subtree AS s
        ON e.manager_id = s.employee_id
)
SELECT
    employee_id,
    employee_name,
    manager_id,
    relative_depth
FROM subtree
ORDER BY relative_depth, employee_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Upward Traversal
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Start at Grace and repeatedly follow the manager relationship.
WITH RECURSIVE management_chain AS (
    SELECT
        employee_id,
        employee_name,
        manager_id,
        0 AS distance
    FROM employee_hierarchy
    WHERE employee_name = 'Grace'

    UNION ALL

    SELECT
        m.employee_id,
        m.employee_name,
        m.manager_id,
        c.distance + 1
    FROM employee_hierarchy AS m
    JOIN management_chain AS c
        ON m.employee_id = c.manager_id
)
SELECT
    employee_id,
    employee_name,
    manager_id,
    distance
FROM management_chain
ORDER BY distance;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Recursive Aggregation
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Derive the complete set of descendants for every employee.
WITH RECURSIVE descendants AS (
    SELECT
        employee_id AS ancestor_id,
        employee_id AS descendant_id
    FROM employee_hierarchy

    UNION ALL

    SELECT
        d.ancestor_id,
        e.employee_id
    FROM descendants AS d
    JOIN employee_hierarchy AS e
        ON e.manager_id = d.descendant_id
)
SELECT
    e.employee_id,
    e.employee_name,
    COUNT(*) - 1 AS descendant_count
FROM descendants AS d
JOIN employee_hierarchy AS e
    ON e.employee_id = d.ancestor_id
GROUP BY
    e.employee_id,
    e.employee_name
ORDER BY e.employee_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Termination
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- A leaf has no matching row for the recursive step.
WITH RECURSIVE hierarchy AS (
    SELECT
        employee_id,
        employee_name,
        manager_id,
        0 AS depth
    FROM employee_hierarchy
    WHERE manager_id IS NULL

    UNION ALL

    SELECT
        e.employee_id,
        e.employee_name,
        e.manager_id,
        h.depth + 1
    FROM employee_hierarchy AS e
    JOIN hierarchy AS h
        ON e.manager_id = h.employee_id
)
SELECT
    MAX(depth) AS maximum_depth,
    COUNT(*) AS employees_reached
FROM hierarchy;


-- Discard the temporary relation and complete experiment.
ROLLBACK;