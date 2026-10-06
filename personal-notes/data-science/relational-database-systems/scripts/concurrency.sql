/*
Concurrency

Demonstrates concurrent transaction behaviour in PostgreSQL using two
independent psql sessions. The examples cover transaction visibility,
READ COMMITTED, REPEATABLE READ, row-level locking, and SERIALIZABLE.

This script is an interactive laboratory rather than a file intended
for execution from beginning to end. Run the marked Session A and
Session B blocks manually in two separate psql terminals.
*/

-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Setup
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Run once before opening the concurrency experiments.
DROP TABLE IF EXISTS concurrency_accounts;

CREATE TABLE concurrency_accounts (
    account_id INTEGER PRIMARY KEY,
    account_name VARCHAR(100) NOT NULL,
    balance NUMERIC(10, 2) NOT NULL
        CHECK (balance >= 0)
);

INSERT INTO concurrency_accounts (
    account_id,
    account_name,
    balance
)
VALUES
    (1, 'Alice', 1000.00),
    (2, 'Bob', 500.00);


SELECT *
FROM concurrency_accounts
ORDER BY account_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- READ COMMITTED: Uncommitted Changes
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Session A
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;

UPDATE concurrency_accounts
SET balance = 900.00
WHERE account_id = 1;

SELECT *
FROM concurrency_accounts
WHERE account_id = 1;

-- Leave this transaction open.


-- Session B
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;

SELECT *
FROM concurrency_accounts
WHERE account_id = 1;

-- Session B still observes the last committed value.


-- Session A
COMMIT;


-- Session B
SELECT *
FROM concurrency_accounts
WHERE account_id = 1;

-- Under READ COMMITTED, the second statement can see the new commit.

COMMIT;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- REPEATABLE READ: Stable Snapshot
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Reset the demonstration state before the next experiment.
UPDATE concurrency_accounts
SET balance = 1000.00
WHERE account_id = 1;


-- Session B
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;

SELECT *
FROM concurrency_accounts
WHERE account_id = 1;

-- Session B establishes a snapshot containing balance = 1000.00.


-- Session A
BEGIN;

UPDATE concurrency_accounts
SET balance = 900.00
WHERE account_id = 1;

COMMIT;


-- Session B
SELECT *
FROM concurrency_accounts
WHERE account_id = 1;

-- The transaction continues to observe balance = 1000.00.

COMMIT;


-- Session B
SELECT *
FROM concurrency_accounts
WHERE account_id = 1;

-- A new transaction context can now observe balance = 900.00.


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Row-Level Locking
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Reset the demonstration state.
UPDATE concurrency_accounts
SET balance = 1000.00
WHERE account_id = 1;


-- Session A
BEGIN;

UPDATE concurrency_accounts
SET balance = balance - 100
WHERE account_id = 1;

-- Leave the transaction open after acquiring the row lock.


-- Session B
BEGIN;

UPDATE concurrency_accounts
SET balance = balance - 50
WHERE account_id = 1;

-- This statement waits because Session A holds a conflicting row lock.


-- Session A
COMMIT;


-- Session B
-- The blocked UPDATE can now complete.

SELECT *
FROM concurrency_accounts
WHERE account_id = 1;

COMMIT;


-- Session B
SELECT *
FROM concurrency_accounts
WHERE account_id = 1;

-- Both committed updates are now visible.


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Explicit Row Lock
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Reset the demonstration state.
UPDATE concurrency_accounts
SET balance = 1000.00
WHERE account_id = 1;


-- Session A
BEGIN;

SELECT *
FROM concurrency_accounts
WHERE account_id = 1
FOR UPDATE;

-- Session A explicitly locks the selected row.


-- Session B
BEGIN;

UPDATE concurrency_accounts
SET balance = 950.00
WHERE account_id = 1;

-- This UPDATE waits for Session A to release the row lock.


-- Session A
COMMIT;


-- Session B
-- The UPDATE can now complete.

COMMIT;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- SERIALIZABLE: Write Skew
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Create a relation for an invariant involving multiple rows.
DROP TABLE IF EXISTS concurrency_doctors;

CREATE TABLE concurrency_doctors (
    doctor_id INTEGER PRIMARY KEY,
    doctor_name VARCHAR(100) NOT NULL,
    on_call BOOLEAN NOT NULL
);

INSERT INTO concurrency_doctors (
    doctor_id,
    doctor_name,
    on_call
)
VALUES
    (1, 'Alice', true),
    (2, 'Bob', true);


-- The intended invariant is that at least one doctor remains on call.
SELECT *
FROM concurrency_doctors
ORDER BY doctor_id;


-- Session A
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;

SELECT COUNT(*) AS doctors_on_call
FROM concurrency_doctors
WHERE on_call = true;

-- Session A observes two doctors on call.


-- Session B
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;

SELECT COUNT(*) AS doctors_on_call
FROM concurrency_doctors
WHERE on_call = true;

-- Session B independently observes the same state.


-- Session A
UPDATE concurrency_doctors
SET on_call = false
WHERE doctor_id = 1;


-- Session B
UPDATE concurrency_doctors
SET on_call = false
WHERE doctor_id = 2;


-- Session A
COMMIT;


-- Session B
COMMIT;

-- PostgreSQL should reject one transaction with a serialization
-- failure rather than allow a non-serializable final state.


-- After resolving the failed transaction, inspect the committed state.
SELECT *
FROM concurrency_doctors
ORDER BY doctor_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Cleanup
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Run only after all experiments have finished.
DROP TABLE concurrency_doctors;
DROP TABLE concurrency_accounts;