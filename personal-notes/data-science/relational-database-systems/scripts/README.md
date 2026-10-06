# SQL Scripts

The scripts in this directory use PostgreSQL and the Northwind sample database.

## Database Setup

Create a local PostgreSQL database named `northwind`:

```sql
CREATE DATABASE northwind;
```

The database schema and sample data are provided in `northwind-setup.sql`, based on the [PostgreSQL version of the Northwind sample database](https://en.wikiversity.org/wiki/Database_Examples/Northwind/PostgreSQL) available from Wikiversity.

From the project directory, populate the database using `psql`:

```bash
psql -U postgres -d northwind -f scripts/northwind_setup.sql
```

Alternatively, connect to the `northwind` database using a PostgreSQL client and execute `northwind-setup.sql` directly.

## Verify the Database

Connect to the database:

```bash
psql -U postgres -d northwind
```

List the available tables:

```text
\dt
```

The database should contain the following tables:

- `categories`
- `customers`
- `employees`
- `orders`
- `orderdetails`
- `products`
- `shippers`
- `suppliers`

A diagram of the database schema is available in `northwind-er-diagram.png`.

## SQL Examples

The remaining SQL scripts contain the examples and experiments developed alongside the technical notes. They assume that the `northwind` database has already been created and populated.