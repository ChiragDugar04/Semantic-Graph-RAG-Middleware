
CREATE DATABASE IF NOT EXISTS middleware_poc;
USE middleware_poc;


CREATE TABLE departments (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(100) NOT NULL UNIQUE,
    budget      DECIMAL(12, 2) NOT NULL,
    location    VARCHAR(100) NOT NULL,
    manager_id  INT NULL
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


CREATE TABLE projects (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    name          VARCHAR(150) NOT NULL UNIQUE,
    description   TEXT,
    budget        DECIMAL(12, 2) NOT NULL,
    start_date    DATE NOT NULL,
    end_date      DATE NULL,
    status        ENUM('planning', 'active', 'completed', 'on_hold')
                  NOT NULL DEFAULT 'active',
    department_id INT NOT NULL,
    manager_id    INT NOT NULL,
    CONSTRAINT fk_project_dept    FOREIGN KEY (department_id)
        REFERENCES departments(id) ON DELETE RESTRICT,
    CONSTRAINT fk_project_manager FOREIGN KEY (manager_id)
        REFERENCES employees(id)   ON DELETE RESTRICT
);


CREATE TABLE project_assignments (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    project_id       INT NOT NULL,
    employee_id      INT NOT NULL,
    role_on_project  VARCHAR(100) NOT NULL,
    assigned_date    DATE NOT NULL,
    CONSTRAINT fk_pa_project  FOREIGN KEY (project_id)
        REFERENCES projects(id)   ON DELETE CASCADE,
    CONSTRAINT fk_pa_employee FOREIGN KEY (employee_id)
        REFERENCES employees(id)  ON DELETE CASCADE,
    CONSTRAINT uq_pa_assignment UNIQUE (project_id, employee_id)
);
