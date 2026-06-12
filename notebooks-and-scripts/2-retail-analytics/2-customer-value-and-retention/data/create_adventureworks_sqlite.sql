-- AdventureWorks OLTP → SQLite starter schema for Customer Analytics
-- Generated from instawdb.sql. SQL Server-only objects (procedures, XML schemas, full-text indexes, triggers) are intentionally omitted.
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS "person_countryregion";
CREATE TABLE "person_countryregion" (
    "CountryRegionCode" TEXT NOT NULL,
    "Name" TEXT NOT NULL,
    "ModifiedDate" TEXT NOT NULL,
    PRIMARY KEY ("CountryRegionCode")
);

DROP TABLE IF EXISTS "sales_salesterritory";
CREATE TABLE "sales_salesterritory" (
    "TerritoryID" INTEGER NOT NULL,
    "Name" TEXT NOT NULL,
    "CountryRegionCode" TEXT NOT NULL,
    "Group" TEXT NOT NULL,
    "SalesYTD" REAL NOT NULL,
    "SalesLastYear" REAL NOT NULL,
    "CostYTD" REAL NOT NULL,
    "CostLastYear" REAL NOT NULL,
    "rowguid" TEXT NOT NULL,
    "ModifiedDate" TEXT NOT NULL,
    PRIMARY KEY ("TerritoryID"),
    FOREIGN KEY ("CountryRegionCode") REFERENCES "person_countryregion" ("CountryRegionCode")
);

DROP TABLE IF EXISTS "person_stateprovince";
CREATE TABLE "person_stateprovince" (
    "StateProvinceID" INTEGER NOT NULL,
    "StateProvinceCode" TEXT NOT NULL,
    "CountryRegionCode" TEXT NOT NULL,
    "IsOnlyStateProvinceFlag" TEXT NOT NULL,
    "Name" TEXT NOT NULL,
    "TerritoryID" INTEGER NOT NULL,
    "rowguid" TEXT NOT NULL,
    "ModifiedDate" TEXT NOT NULL,
    PRIMARY KEY ("StateProvinceID"),
    FOREIGN KEY ("CountryRegionCode") REFERENCES "person_countryregion" ("CountryRegionCode"),
    FOREIGN KEY ("TerritoryID") REFERENCES "sales_salesterritory" ("TerritoryID")
);

DROP TABLE IF EXISTS "person_address";
CREATE TABLE "person_address" (
    "AddressID" INTEGER NOT NULL,
    "AddressLine1" TEXT NOT NULL,
    "AddressLine2" TEXT,
    "City" TEXT NOT NULL,
    "StateProvinceID" INTEGER NOT NULL,
    "PostalCode" TEXT NOT NULL,
    "SpatialLocation" TEXT,
    "rowguid" TEXT NOT NULL,
    "ModifiedDate" TEXT NOT NULL,
    PRIMARY KEY ("AddressID"),
    FOREIGN KEY ("StateProvinceID") REFERENCES "person_stateprovince" ("StateProvinceID")
);

DROP TABLE IF EXISTS "person_addresstype";
CREATE TABLE "person_addresstype" (
    "AddressTypeID" INTEGER NOT NULL,
    "Name" TEXT NOT NULL,
    "rowguid" TEXT NOT NULL,
    "ModifiedDate" TEXT NOT NULL,
    PRIMARY KEY ("AddressTypeID")
);

DROP TABLE IF EXISTS "person_businessentityaddress";
CREATE TABLE "person_businessentityaddress" (
    "BusinessEntityID" INTEGER NOT NULL,
    "AddressID" INTEGER NOT NULL,
    "AddressTypeID" INTEGER NOT NULL,
    "rowguid" TEXT NOT NULL,
    "ModifiedDate" TEXT NOT NULL,
    PRIMARY KEY ("BusinessEntityID", "AddressID", "AddressTypeID"),
    FOREIGN KEY ("AddressID") REFERENCES "person_address" ("AddressID"),
    FOREIGN KEY ("AddressTypeID") REFERENCES "person_addresstype" ("AddressTypeID")
);

DROP TABLE IF EXISTS "person_person";
CREATE TABLE "person_person" (
    "BusinessEntityID" INTEGER NOT NULL,
    "PersonType" TEXT NOT NULL,
    "NameStyle" TEXT NOT NULL,
    "Title" TEXT,
    "FirstName" TEXT NOT NULL,
    "MiddleName" TEXT,
    "LastName" TEXT NOT NULL,
    "Suffix" TEXT,
    "EmailPromotion" INTEGER NOT NULL,
    "AdditionalContactInfo" TEXT,
    "Demographics" TEXT,
    "rowguid" TEXT NOT NULL,
    "ModifiedDate" TEXT NOT NULL,
    PRIMARY KEY ("BusinessEntityID")
);

DROP TABLE IF EXISTS "person_emailaddress";
CREATE TABLE "person_emailaddress" (
    "BusinessEntityID" INTEGER NOT NULL,
    "EmailAddressID" INTEGER NOT NULL,
    "EmailAddress" TEXT,
    "rowguid" TEXT NOT NULL,
    "ModifiedDate" TEXT NOT NULL,
    PRIMARY KEY ("BusinessEntityID", "EmailAddressID"),
    FOREIGN KEY ("BusinessEntityID") REFERENCES "person_person" ("BusinessEntityID")
);

DROP TABLE IF EXISTS "sales_store";
CREATE TABLE "sales_store" (
    "BusinessEntityID" INTEGER NOT NULL,
    "Name" TEXT NOT NULL,
    "SalesPersonID" INTEGER,
    "Demographics" TEXT,
    "rowguid" TEXT NOT NULL,
    "ModifiedDate" TEXT NOT NULL,
    PRIMARY KEY ("BusinessEntityID")
);

DROP TABLE IF EXISTS "sales_customer";
CREATE TABLE "sales_customer" (
    "CustomerID" INTEGER NOT NULL,
    "PersonID" INTEGER,
    "StoreID" INTEGER,
    "TerritoryID" INTEGER,
    "AccountNumber" TEXT,
    "rowguid" TEXT NOT NULL,
    "ModifiedDate" TEXT NOT NULL,
    PRIMARY KEY ("CustomerID"),
    FOREIGN KEY ("PersonID") REFERENCES "person_person" ("BusinessEntityID"),
    FOREIGN KEY ("StoreID") REFERENCES "sales_store" ("BusinessEntityID"),
    FOREIGN KEY ("TerritoryID") REFERENCES "sales_salesterritory" ("TerritoryID")
);

DROP TABLE IF EXISTS "purchasing_shipmethod";
CREATE TABLE "purchasing_shipmethod" (
    "ShipMethodID" INTEGER NOT NULL,
    "Name" TEXT NOT NULL,
    "ShipBase" REAL NOT NULL,
    "ShipRate" REAL NOT NULL,
    "rowguid" TEXT NOT NULL,
    "ModifiedDate" TEXT NOT NULL,
    PRIMARY KEY ("ShipMethodID")
);

DROP TABLE IF EXISTS "sales_specialoffer";
CREATE TABLE "sales_specialoffer" (
    "SpecialOfferID" INTEGER NOT NULL,
    "Description" TEXT NOT NULL,
    "DiscountPct" REAL NOT NULL,
    "Type" TEXT NOT NULL,
    "Category" TEXT NOT NULL,
    "StartDate" TEXT NOT NULL,
    "EndDate" TEXT NOT NULL,
    "MinQty" INTEGER NOT NULL,
    "MaxQty" INTEGER,
    "rowguid" TEXT NOT NULL,
    "ModifiedDate" TEXT NOT NULL,
    PRIMARY KEY ("SpecialOfferID")
);

DROP TABLE IF EXISTS "sales_specialofferproduct";
CREATE TABLE "sales_specialofferproduct" (
    "SpecialOfferID" INTEGER NOT NULL,
    "ProductID" INTEGER NOT NULL,
    "rowguid" TEXT NOT NULL,
    "ModifiedDate" TEXT NOT NULL,
    PRIMARY KEY ("SpecialOfferID", "ProductID"),
    FOREIGN KEY ("SpecialOfferID") REFERENCES "sales_specialoffer" ("SpecialOfferID"),
    FOREIGN KEY ("ProductID") REFERENCES "production_product" ("ProductID")
);

DROP TABLE IF EXISTS "production_productcategory";
CREATE TABLE "production_productcategory" (
    "ProductCategoryID" INTEGER NOT NULL,
    "Name" TEXT NOT NULL,
    "rowguid" TEXT NOT NULL,
    "ModifiedDate" TEXT NOT NULL,
    PRIMARY KEY ("ProductCategoryID")
);

DROP TABLE IF EXISTS "production_productsubcategory";
CREATE TABLE "production_productsubcategory" (
    "ProductSubcategoryID" INTEGER NOT NULL,
    "ProductCategoryID" INTEGER NOT NULL,
    "Name" TEXT NOT NULL,
    "rowguid" TEXT NOT NULL,
    "ModifiedDate" TEXT NOT NULL,
    PRIMARY KEY ("ProductSubcategoryID"),
    FOREIGN KEY ("ProductCategoryID") REFERENCES "production_productcategory" ("ProductCategoryID")
);

DROP TABLE IF EXISTS "production_product";
CREATE TABLE "production_product" (
    "ProductID" INTEGER NOT NULL,
    "Name" TEXT NOT NULL,
    "ProductNumber" TEXT NOT NULL,
    "MakeFlag" TEXT NOT NULL,
    "FinishedGoodsFlag" TEXT NOT NULL,
    "Color" TEXT,
    "SafetyStockLevel" INTEGER NOT NULL,
    "ReorderPoint" INTEGER NOT NULL,
    "StandardCost" REAL NOT NULL,
    "ListPrice" REAL NOT NULL,
    "Size" TEXT,
    "SizeUnitMeasureCode" TEXT,
    "WeightUnitMeasureCode" TEXT,
    "Weight" REAL,
    "DaysToManufacture" INTEGER NOT NULL,
    "ProductLine" TEXT,
    "Class" TEXT,
    "Style" TEXT,
    "ProductSubcategoryID" INTEGER,
    "ProductModelID" INTEGER,
    "SellStartDate" TEXT NOT NULL,
    "SellEndDate" TEXT,
    "DiscontinuedDate" TEXT,
    "rowguid" TEXT NOT NULL,
    "ModifiedDate" TEXT NOT NULL,
    PRIMARY KEY ("ProductID"),
    FOREIGN KEY ("ProductSubcategoryID") REFERENCES "production_productsubcategory" ("ProductSubcategoryID")
);

DROP TABLE IF EXISTS "sales_salesorderheader";
CREATE TABLE "sales_salesorderheader" (
    "SalesOrderID" INTEGER NOT NULL,
    "RevisionNumber" INTEGER NOT NULL,
    "OrderDate" TEXT NOT NULL,
    "DueDate" TEXT NOT NULL,
    "ShipDate" TEXT,
    "Status" INTEGER NOT NULL,
    "OnlineOrderFlag" TEXT NOT NULL,
    "SalesOrderNumber" TEXT,
    "PurchaseOrderNumber" TEXT,
    "AccountNumber" TEXT,
    "CustomerID" INTEGER NOT NULL,
    "SalesPersonID" INTEGER,
    "TerritoryID" INTEGER,
    "BillToAddressID" INTEGER NOT NULL,
    "ShipToAddressID" INTEGER NOT NULL,
    "ShipMethodID" INTEGER NOT NULL,
    "CreditCardID" INTEGER,
    "CreditCardApprovalCode" TEXT,
    "CurrencyRateID" INTEGER,
    "SubTotal" REAL NOT NULL,
    "TaxAmt" REAL NOT NULL,
    "Freight" REAL NOT NULL,
    "TotalDue" REAL,
    "Comment" TEXT,
    "rowguid" TEXT NOT NULL,
    "ModifiedDate" TEXT NOT NULL,
    PRIMARY KEY ("SalesOrderID"),
    FOREIGN KEY ("CustomerID") REFERENCES "sales_customer" ("CustomerID"),
    FOREIGN KEY ("BillToAddressID") REFERENCES "person_address" ("AddressID"),
    FOREIGN KEY ("ShipToAddressID") REFERENCES "person_address" ("AddressID"),
    FOREIGN KEY ("ShipMethodID") REFERENCES "purchasing_shipmethod" ("ShipMethodID"),
    FOREIGN KEY ("TerritoryID") REFERENCES "sales_salesterritory" ("TerritoryID")
);

DROP TABLE IF EXISTS "sales_salesorderdetail";
CREATE TABLE "sales_salesorderdetail" (
    "SalesOrderID" INTEGER NOT NULL,
    "SalesOrderDetailID" INTEGER NOT NULL,
    "CarrierTrackingNumber" TEXT,
    "OrderQty" INTEGER NOT NULL,
    "ProductID" INTEGER NOT NULL,
    "SpecialOfferID" INTEGER NOT NULL,
    "UnitPrice" REAL NOT NULL,
    "UnitPriceDiscount" REAL NOT NULL,
    "LineTotal" REAL,
    "rowguid" TEXT NOT NULL,
    "ModifiedDate" TEXT NOT NULL,
    PRIMARY KEY ("SalesOrderID", "SalesOrderDetailID"),
    FOREIGN KEY ("SalesOrderID") REFERENCES "sales_salesorderheader" ("SalesOrderID"),
    FOREIGN KEY ("SpecialOfferID", "ProductID") REFERENCES "sales_specialofferproduct" ("SpecialOfferID", "ProductID"),
    FOREIGN KEY ("ProductID") REFERENCES "production_product" ("ProductID")
);

CREATE INDEX IF NOT EXISTS "idx_soh_customer_orderdate" ON "sales_salesorderheader" ("CustomerID", "OrderDate");
CREATE INDEX IF NOT EXISTS "idx_sod_product" ON "sales_salesorderdetail" ("ProductID");
CREATE INDEX IF NOT EXISTS "idx_product_subcategory" ON "production_product" ("ProductSubcategoryID");
CREATE INDEX IF NOT EXISTS "idx_customer_person" ON "sales_customer" ("PersonID");
CREATE INDEX IF NOT EXISTS "idx_customer_territory" ON "sales_customer" ("TerritoryID");