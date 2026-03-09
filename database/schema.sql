
CREATE DATABASE IF NOT EXISTS middleware_poc;
USE middleware_poc;

-- Drop tables in reverse FK order if re-running
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS departments;


CREATE TABLE departments (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(100) NOT NULL UNIQUE,
    budget      DECIMAL(12, 2) NOT NULL,
    location    VARCHAR(100) NOT NULL,
    manager_id  INT NULL   -- FK to employees, set via ALTER below
);


CREATE TABLE employees (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    department_id   INT NOT NULL,
    salary          DECIMAL(10, 2) NOT NULL,
    hire_date       DATE NOT NULL,
    role            VARCHAR(100) NOT NULL,
    email           VARCHAR(150) NOT NULL UNIQUE,
    CONSTRAINT fk_emp_dept FOREIGN KEY (department_id)
        REFERENCES departments(id)
        ON DELETE RESTRICT
        ON UPDATE CASCADE
);

-- Now add the manager FK on departments (circular reference resolved)
ALTER TABLE departments
    ADD CONSTRAINT fk_dept_manager
    FOREIGN KEY (manager_id) REFERENCES employees(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE;


CREATE TABLE products (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(150) NOT NULL UNIQUE,
    category        VARCHAR(100) NOT NULL,
    price           DECIMAL(10, 2) NOT NULL,
    stock_quantity  INT NOT NULL DEFAULT 0,
    supplier        VARCHAR(150) NOT NULL
);


CREATE TABLE orders (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    product_id      INT NOT NULL,
    employee_id     INT NOT NULL,
    quantity        INT NOT NULL,
    order_date      DATE NOT NULL,
    status          ENUM('pending', 'processing', 'shipped', 'delivered', 'cancelled')
                    NOT NULL DEFAULT 'pending',
    CONSTRAINT fk_order_product  FOREIGN KEY (product_id)
        REFERENCES products(id) ON DELETE RESTRICT,
    CONSTRAINT fk_order_employee FOREIGN KEY (employee_id)
        REFERENCES employees(id) ON DELETE RESTRICT
);
