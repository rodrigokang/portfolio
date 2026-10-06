/*
Semistructured Data

Demonstrates semistructured data in PostgreSQL using JSONB alongside
ordinary relational attributes. The examples cover JSON documents,
nested values, arrays, extraction operators, containment predicates,
document expansion, and updates.

A temporary relation is used so that the Northwind schema remains
unchanged.
*/

BEGIN;

-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- JSONB Documents
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Combine fixed relational attributes with flexible JSONB metadata.
CREATE TEMPORARY TABLE customer_profiles (
    customer_id INTEGER PRIMARY KEY,
    customer_name VARCHAR(100) NOT NULL,
    profile JSONB NOT NULL
);


-- Insert documents whose internal structures need not be identical.
INSERT INTO customer_profiles (
    customer_id,
    customer_name,
    profile
)
VALUES
    (
        1,
        'Alice',
        '{
            "segment": "premium",
            "preferences": {
                "language": "English",
                "currency": "NZD"
            },
            "interests": ["analytics", "finance"],
            "active": true
        }'
    ),
    (
        2,
        'Bob',
        '{
            "segment": "standard",
            "preferences": {
                "language": "Spanish"
            },
            "interests": ["travel"],
            "active": true
        }'
    ),
    (
        3,
        'Carol',
        '{
            "segment": "premium",
            "preferences": {
                "language": "English",
                "currency": "USD"
            },
            "interests": ["analytics", "technology", "travel"],
            "active": false,
            "loyalty": {
                "level": "gold",
                "points": 4200
            }
        }'
    );


-- Inspect relational attributes and JSONB documents together.
SELECT
    customer_id,
    customer_name,
    profile
FROM customer_profiles
ORDER BY customer_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Value Extraction
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- The -> operator returns a JSON value.
SELECT
    customer_name,
    profile -> 'segment' AS segment
FROM customer_profiles
ORDER BY customer_id;


-- The ->> operator extracts the value as SQL text.
SELECT
    customer_name,
    profile ->> 'segment' AS segment
FROM customer_profiles
ORDER BY customer_id;


-- Navigate through nested JSON objects.
SELECT
    customer_name,
    profile -> 'preferences' ->> 'language' AS language,
    profile -> 'preferences' ->> 'currency' AS currency
FROM customer_profiles
ORDER BY customer_id;


-- Missing attributes produce NULL rather than requiring one schema.
SELECT
    customer_name,
    profile -> 'loyalty' ->> 'level' AS loyalty_level
FROM customer_profiles
ORDER BY customer_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- JSON Predicates
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Filter rows using a scalar value stored inside the document.
SELECT
    customer_id,
    customer_name
FROM customer_profiles
WHERE profile ->> 'segment' = 'premium'
ORDER BY customer_id;


-- Test whether a JSONB document contains a specified structure.
SELECT
    customer_id,
    customer_name
FROM customer_profiles
WHERE profile @> '{"active": true}'
ORDER BY customer_id;


-- Containment can also test nested document structure.
SELECT
    customer_id,
    customer_name
FROM customer_profiles
WHERE profile @> '{
    "preferences": {
        "language": "English"
    }
}'
ORDER BY customer_id;


-- Test whether an array contains a specified element.
SELECT
    customer_id,
    customer_name
FROM customer_profiles
WHERE profile -> 'interests' @> '["analytics"]'
ORDER BY customer_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Array Expansion
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Expand each JSON array into a set of relational rows.
SELECT
    c.customer_id,
    c.customer_name,
    interest
FROM customer_profiles AS c
CROSS JOIN LATERAL jsonb_array_elements_text(
    c.profile -> 'interests'
) AS interest
ORDER BY c.customer_id, interest;


-- Aggregate the expanded values using ordinary relational operations.
SELECT
    interest,
    COUNT(*) AS customer_count
FROM customer_profiles AS c
CROSS JOIN LATERAL jsonb_array_elements_text(
    c.profile -> 'interests'
) AS interest
GROUP BY interest
ORDER BY customer_count DESC, interest;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Relational and Semistructured Data
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Relational and JSON attributes can participate in one query.
SELECT
    customer_id,
    customer_name,
    profile ->> 'segment' AS segment,
    profile -> 'preferences' ->> 'language' AS language
FROM customer_profiles
WHERE customer_id >= 2
  AND profile ->> 'segment' = 'premium'
ORDER BY customer_id;


-- Derived JSON values can also be grouped relationally.
SELECT
    profile ->> 'segment' AS segment,
    COUNT(*) AS customer_count
FROM customer_profiles
GROUP BY profile ->> 'segment'
ORDER BY segment;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Document Updates
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

SAVEPOINT json_update_test;


-- Replace one value inside a JSONB document.
UPDATE customer_profiles
SET profile = jsonb_set(
    profile,
    '{preferences,currency}',
    '"AUD"',
    true
)
WHERE customer_id = 1;


-- Inspect the modified nested value.
SELECT
    customer_id,
    customer_name,
    profile -> 'preferences' ->> 'currency' AS currency
FROM customer_profiles
WHERE customer_id = 1;


-- Add an attribute that was not present in the original document.
UPDATE customer_profiles
SET profile = jsonb_set(
    profile,
    '{loyalty}',
    '{"level": "silver", "points": 1200}',
    true
)
WHERE customer_id = 2;


-- The document can evolve without altering the table definition.
SELECT
    customer_id,
    customer_name,
    profile
FROM customer_profiles
WHERE customer_id = 2;


-- Restore the original documents.
ROLLBACK TO SAVEPOINT json_update_test;


-- Verify that the temporary modifications were discarded.
SELECT
    customer_id,
    customer_name,
    profile
FROM customer_profiles
ORDER BY customer_id;


-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
-- Document Structure
-- <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

-- Inspect the JSON type associated with selected document values.
SELECT
    customer_name,
    jsonb_typeof(profile) AS profile_type,
    jsonb_typeof(profile -> 'preferences') AS preferences_type,
    jsonb_typeof(profile -> 'interests') AS interests_type,
    jsonb_typeof(profile -> 'active') AS active_type
FROM customer_profiles
ORDER BY customer_id;


-- List the top-level keys present in each document.
SELECT
    c.customer_id,
    c.customer_name,
    key
FROM customer_profiles AS c
CROSS JOIN LATERAL jsonb_object_keys(c.profile) AS key
ORDER BY c.customer_id, key;


-- Discard the temporary relation and complete experiment.
ROLLBACK;