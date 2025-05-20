# Using SQLAlchemy ORM with an Existing Oracle Database

This guide covers how to use SQLAlchemy ORM to interact with an existing Oracle database when you only need to work with a subset of tables.

## Table of Contents
- [Installation](#installation)
- [Connection Setup](#connection-setup)
- [Defining Models](#defining-models)
  - [Manual Model Definition](#manual-model-definition)
  - [Reflecting Existing Tables](#reflecting-existing-tables)
- [Working with Data](#working-with-data)
  - [Basic CRUD Operations](#basic-crud-operations)
  - [Common Query Patterns](#common-query-patterns)
  - [Relationships](#relationships)
- [Advanced Usage](#advanced-usage)
  - [Working with Raw SQL](#working-with-raw-sql)
  - [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Installation

First, install SQLAlchemy and the Oracle database adapter:

```bash
pip install sqlalchemy cx_Oracle
```

You'll also need the Oracle Instant Client libraries installed on your system.

## Connection Setup

Create a connection to your Oracle database using SQLAlchemy's engine:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Format: oracle+cx_oracle://username:password@hostname:port/service_name
DATABASE_URI = 'oracle+cx_oracle://user:pass@hostname:1521/service'

# Create engine
engine = create_engine(DATABASE_URI, echo=False)

# Create session factory
Session = sessionmaker(bind=engine)
```

## Defining Models

### Manual Model Definition

Manually define only the tables you need for your application:

```python
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Customer(Base):
    __tablename__ = 'customers'  # Must match the actual table name in Oracle
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    email = Column(String(100))
    
    # Define relationship to Order model
    orders = relationship("Order", back_populates="customer")
    
    def __repr__(self):
        return f"<Customer(name='{self.name}', email='{self.email}')>"

class Order(Base):
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'))
    order_date = Column(DateTime)
    total_amount = Column(Float)
    
    # Define relationship to Customer model
    customer = relationship("Customer", back_populates="orders")
    
    def __repr__(self):
        return f"<Order(id={self.id}, date='{self.order_date}', amount={self.total_amount})>"
```

### Reflecting Existing Tables

If you prefer not to manually define the models, you can use SQLAlchemy's reflection capabilities:

#### Option 1: Reflect specific tables (Core approach)

```python
from sqlalchemy import MetaData, Table

metadata = MetaData()
# Only reflect the tables you need
customers = Table('customers', metadata, autoload_with=engine)
orders = Table('orders', metadata, autoload_with=engine)

# Now you can use these tables with SQLAlchemy Core
```

#### Option 2: Automap for ORM access

```python
from sqlalchemy.ext.automap import automap_base

Base = automap_base()
# Reflect all tables
Base.prepare(engine, reflect=True)

# Access only the tables you need as ORM models
Customers = Base.classes.customers
Orders = Base.classes.orders
```

#### Option 3: Selective reflection with declarative models

```python
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Customer(Base):
    __tablename__ = 'customers'
    __table_args__ = {'autoload_with': engine}
    
    # You can add relationships or methods here
    orders = relationship("Order", back_populates="customer")

class Order(Base):
    __tablename__ = 'orders'
    __table_args__ = {'autoload_with': engine}
    
    customer = relationship("Customer", back_populates="orders")
```

## Working with Data

### Basic CRUD Operations

Create a session and perform basic operations:

```python
# Create a session
session = Session()

# CREATE
new_customer = Customer(name="Jane Doe", email="jane@example.com")
session.add(new_customer)
session.commit()

# READ
customer = session.query(Customer).filter_by(name="Jane Doe").first()
print(customer.name, customer.email)

# UPDATE
customer.email = "jane.doe@example.com"
session.commit()

# DELETE
session.delete(customer)
session.commit()

# Don't forget to close your session when done
session.close()
```

### Common Query Patterns

```python
# Basic filtering
customers = session.query(Customer).filter(Customer.name.like('J%')).all()

# Joins
orders_with_customers = session.query(Order, Customer).\
    join(Customer, Order.customer_id == Customer.id).\
    filter(Order.total_amount > 100).\
    all()

# Aggregations
from sqlalchemy import func
total_sales = session.query(func.sum(Order.total_amount)).scalar()

# Pagination
page_size = 10
page_number = 2
customers = session.query(Customer).\
    order_by(Customer.name).\
    limit(page_size).\
    offset((page_number - 1) * page_size).\
    all()
```

### Relationships

Working with related models:

```python
# Access related orders from a customer
customer = session.query(Customer).filter_by(id=1).first()
for order in customer.orders:
    print(f"Order #{order.id}: ${order.total_amount} on {order.order_date}")

# Create a new order for existing customer
new_order = Order(order_date=datetime.now(), total_amount=250.00)
customer.orders.append(new_order)  # SQLAlchemy will handle setting customer_id
session.commit()

# Query with joins using relationships
high_value_customers = session.query(Customer).\
    join(Customer.orders).\
    filter(Order.total_amount > 1000).\
    distinct().\
    all()
```

## Advanced Usage

### Working with Raw SQL

For complex queries or Oracle-specific features:

```python
from sqlalchemy import text

# Execute raw SQL with parameters
result = session.execute(
    text("SELECT * FROM customers WHERE name LIKE :name_pattern"),
    {"name_pattern": "J%"}
)
for row in result:
    print(row)

# Use SQL expressions
from sqlalchemy.sql import select
stmt = select([Customer]).where(Customer.name == 'John Smith')
result = session.execute(stmt)
```

### Performance Considerations

```python
# Eager loading for relationships
customers = session.query(Customer).\
    options(joinedload(Customer.orders)).\
    filter(Customer.name.like('J%')).\
    all()

# For large result sets, use yield_per
for customer in session.query(Customer).yield_per(100):
    # Process each customer, loaded in batches of 100
    process_customer(customer)
```

## Troubleshooting

### Common Oracle-Specific Issues

1. **Oracle Identifier Case-Sensitivity**:
   Oracle stores identifiers in uppercase by default. If your models use lowercase table names but Oracle has them in uppercase, you might need to specify:
   ```python
   __tablename__ = 'CUSTOMERS'  # Uppercase to match Oracle's default behavior
   ```
   
   Alternatively, you can configure SQLAlchemy to handle this automatically:
   ```python
   engine = create_engine(DATABASE_URI, case_sensitive=False)
   ```

2. **Oracle Reserved Words**:
   If a table or column name is an Oracle reserved word, use:
   ```python
   from sqlalchemy import Column, Table
   
   # For a column
   my_table = Table('mytable', metadata,
                   Column('select', Integer, key='select_'),
                   ...)
   
   # Or in a model
   class MyModel(Base):
       __tablename__ = 'mytable'
       select_ = Column('select', Integer)
   ```

3. **Date and Time Handling**:
   Oracle has specific date/time types. Use SQLAlchemy's types:
   ```python
   from sqlalchemy import DateTime, Date
   
   class MyModel(Base):
       __tablename__ = 'mytable'
       created_at = Column(DateTime)
       created_date = Column(Date)
   ```

4. **Connection Reset Issues**:
   If you encounter connection resets with Oracle, consider configuring the engine with a connection pool:
   ```python
   engine = create_engine(
       DATABASE_URI,
       pool_size=10,
       max_overflow=20,
       pool_recycle=3600  # Recycle connections after 1 hour
   )
   ```
