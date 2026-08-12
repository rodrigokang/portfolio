-- create_and_populate.sql

-- Create the "users" table
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER
);

-- Insert data into the "users" table
INSERT INTO users (name, age) VALUES ('John', 30);
INSERT INTO users (name, age) VALUES ('Mary', 25);
INSERT INTO users (name, age) VALUES ('Peter', 40);