/*
Transactions

Demonstrates transaction management in PostgreSQL using temporary
tables. The examples cover COMMIT, ROLLBACK, atomic updates,
savepoints, statement errors, and transaction isolation levels.

A temporary accounts relation is used so that the Northwind schema
remains unchanged.
*/

-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Test Relation
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

CREATE TEMPORARY TABLE demo_accounts (
    account_id INTEGER PRIMARY KEY,
    account_name VARCHAR(100) NOT NULL,
    balance NUMERIC(10, 2) NOT NULL
        CHECK (balance >= 0)
);


INSERT INTO demo_accounts (
    account_id,
    account_name,
    balance
)
VALUES
    (1, 'Alice', 1000.00),
    (2, 'Bob', 500.00),
    (3, 'Carol', 750.00);


SELECT *
FROM demo_accounts
ORDER BY account_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- COMMIT
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Execute two related changes as one transaction.
BEGIN;

UPDATE demo_accounts
SET balance = balance - 100
WHERE account_id = 1;

UPDATE demo_accounts
SET balance = balance + 100
WHERE account_id = 2;


-- Both changes are visible inside the current transaction.
SELECT *
FROM demo_accounts
ORDER BY account_id;


-- Make the transaction durable from the session's perspective.
COMMIT;


-- The committed state remains visible after the transaction ends.
SELECT *
FROM demo_accounts
ORDER BY account_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- ROLLBACK
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

BEGIN;


-- Perform another transfer.
UPDATE demo_accounts
SET balance = balance - 200
WHERE account_id = 2;

UPDATE demo_accounts
SET balance = balance + 200
WHERE account_id = 3;


-- Observe the modified state before ending the transaction.
SELECT *
FROM demo_accounts
ORDER BY account_id;


-- Discard every change made since BEGIN.
ROLLBACK;


-- The relation returns to its last committed state.
SELECT *
FROM demo_accounts
ORDER BY account_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Atomicity
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

BEGIN;


-- Debit one account as the first part of a transfer.
UPDATE demo_accounts
SET balance = balance - 300
WHERE account_id = 1;


-- Inspect the intermediate state inside the transaction.
SELECT *
FROM demo_accounts
ORDER BY account_id;


-- Abort the unit of work before completing the transfer.
ROLLBACK;


-- The partial transfer does not remain in the database.
SELECT *
FROM demo_accounts
ORDER BY account_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Savepoints
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

BEGIN;


-- Perform an initial valid operation.
UPDATE demo_accounts
SET balance = balance + 50
WHERE account_id = 3;


SAVEPOINT transfer_test;


-- Perform additional changes after the savepoint.
UPDATE demo_accounts
SET balance = balance - 150
WHERE account_id = 1;

UPDATE demo_accounts
SET balance = balance + 150
WHERE account_id = 2;


SELECT *
FROM demo_accounts
ORDER BY account_id;


-- Undo only the work performed after the savepoint.
ROLLBACK TO SAVEPOINT transfer_test;


-- The earlier update remains part of the current transaction.
SELECT *
FROM demo_accounts
ORDER BY account_id;


-- Discard the entire transaction, including the earlier update.
ROLLBACK;


-- Return to the last committed state.
SELECT *
FROM demo_accounts
ORDER BY account_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Constraint Failure
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

BEGIN;

SAVEPOINT invalid_transfer;


-- This debit violates the CHECK constraint on balance.
UPDATE demo_accounts
SET balance = balance - 5000
WHERE account_id = 1;


-- Recover the transaction after the failed statement.
ROLLBACK TO SAVEPOINT invalid_transfer;


-- The transaction is usable again after rollback to the savepoint.
UPDATE demo_accounts
SET balance = balance - 50
WHERE account_id = 1;

UPDATE demo_accounts
SET balance = balance + 50
WHERE account_id = 2;


SELECT *
FROM demo_accounts
ORDER BY account_id;


-- Discard the complete demonstration transaction.
ROLLBACK;


-- Confirm that the failed and valid updates were both discarded.
SELECT *
FROM demo_accounts
ORDER BY account_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Transaction Isolation
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- PostgreSQL reports the default isolation level for the session.
SHOW default_transaction_isolation;


-- Set the isolation level for one transaction.
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;

SHOW transaction_isolation;

SELECT *
FROM demo_accounts
ORDER BY account_id;

COMMIT;


-- Execute another transaction under REPEATABLE READ.
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;

SHOW transaction_isolation;

SELECT *
FROM demo_accounts
ORDER BY account_id;

COMMIT;


-- Execute a transaction under SERIALIZABLE isolation.
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;

SHOW transaction_isolation;

SELECT *
FROM demo_accounts
ORDER BY account_id;

COMMIT;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Final State
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Only the transfer committed in the first example remains.
SELECT *
FROM demo_accounts
ORDER BY account_id;