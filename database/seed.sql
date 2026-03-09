USE middleware_poc;


INSERT INTO departments (name, budget, location, manager_id) VALUES
('Engineering',     1200000.00, 'Building A - Floor 3',  NULL),
('Sales',            850000.00, 'Building B - Floor 1',  NULL),
('Human Resources',  400000.00, 'Building A - Floor 1',  NULL),
('Marketing',        600000.00, 'Building B - Floor 2',  NULL),
('Operations',       750000.00, 'Building C - Floor 1',  NULL);


INSERT INTO employees (name, department_id, salary, hire_date, role, email) VALUES
-- Engineering (dept 1)
('Sarah Connor',    1, 95000.00, '2021-03-15', 'Senior Engineer',      'sarah.connor@company.com'),
('John Reese',      1, 88000.00, '2022-06-01', 'Software Engineer',    'john.reese@company.com'),
('Alan Turing',     1, 115000.00,'2019-01-10', 'Lead Engineer',        'alan.turing@company.com'),
('Grace Hopper',    1, 105000.00,'2020-07-22', 'Principal Engineer',   'grace.hopper@company.com'),
('Linus Torvalds',  1, 130000.00,'2018-04-05', 'Engineering Manager',  'linus.torvalds@company.com'),
-- Sales (dept 2)
('Michael Scott',   2, 72000.00, '2020-09-14', 'Sales Manager',        'michael.scott@company.com'),
('Jim Halpert',     2, 65000.00, '2021-11-03', 'Sales Representative', 'jim.halpert@company.com'),
('Dwight Schrute',  2, 67000.00, '2021-05-19', 'Senior Sales Rep',     'dwight.schrute@company.com'),
('Pam Beesly',      2, 58000.00, '2022-02-28', 'Sales Coordinator',    'pam.beesly@company.com'),
-- HR (dept 3)
('Toby Flenderson', 3, 62000.00, '2019-08-12', 'HR Manager',           'toby.flenderson@company.com'),
('Kelly Kapoor',    3, 55000.00, '2022-04-07', 'HR Specialist',        'kelly.kapoor@company.com'),
-- Marketing (dept 4)
('Don Draper',      4, 98000.00, '2020-01-20', 'Marketing Director',   'don.draper@company.com'),
('Peggy Olson',     4, 82000.00, '2021-08-30', 'Senior Copywriter',    'peggy.olson@company.com'),
('Roger Sterling',  4, 88000.00, '2020-06-15', 'Brand Manager',        'roger.sterling@company.com'),
-- Operations (dept 5)
('Leslie Knope',    5, 78000.00, '2019-11-01', 'Operations Manager',   'leslie.knope@company.com'),
('Ben Wyatt',       5, 74000.00, '2020-03-17', 'Operations Analyst',   'ben.wyatt@company.com'),
('Tom Haverford',   5, 60000.00, '2022-07-11', 'Operations Coordinator','tom.haverford@company.com');


UPDATE departments SET manager_id = (SELECT id FROM employees WHERE name = 'Linus Torvalds') WHERE name = 'Engineering';
UPDATE departments SET manager_id = (SELECT id FROM employees WHERE name = 'Michael Scott')  WHERE name = 'Sales';
UPDATE departments SET manager_id = (SELECT id FROM employees WHERE name = 'Toby Flenderson') WHERE name = 'Human Resources';
UPDATE departments SET manager_id = (SELECT id FROM employees WHERE name = 'Don Draper')     WHERE name = 'Marketing';
UPDATE departments SET manager_id = (SELECT id FROM employees WHERE name = 'Leslie Knope')   WHERE name = 'Operations';


INSERT INTO products (name, category, price, stock_quantity, supplier) VALUES
('Laptop Pro 15',       'Electronics',  1299.99,  45, 'TechSupply Co.'),
('Wireless Mouse',      'Electronics',    29.99, 230, 'PeripheralPlus'),
('Standing Desk',       'Furniture',     599.00,  18, 'OfficeFit Ltd.'),
('Ergonomic Chair',     'Furniture',     449.00,  27, 'OfficeFit Ltd.'),
('USB-C Hub',           'Electronics',    49.99, 175, 'TechSupply Co.'),
('Mechanical Keyboard', 'Electronics',    89.99,  92, 'PeripheralPlus'),
('Monitor 27 inch',     'Electronics',   399.99,  33, 'TechSupply Co.'),
('Whiteboard 4x6',      'Office Supply',  79.99,  12, 'OfficeWorld'),
('Notebook Pack',       'Office Supply',   9.99, 500, 'OfficeWorld'),
('Webcam HD',           'Electronics',    79.99,  68, 'PeripheralPlus'),
('Desk Lamp',           'Furniture',      34.99, 145, 'OfficeFit Ltd.'),
('Printer Paper A4',    'Office Supply',  24.99, 820, 'OfficeWorld');


INSERT INTO orders (product_id, employee_id, quantity, order_date, status) VALUES
(1,  15, 3, '2024-11-02', 'delivered'),
(2,  7,  10,'2024-11-10', 'delivered'),
(3,  10, 2, '2024-11-15', 'delivered'),
(4,  1,  1, '2024-11-20', 'delivered'),
(5,  3,  5, '2024-12-01', 'delivered'),
(6,  2,  4, '2024-12-05', 'shipped'),
(7,  12, 2, '2024-12-10', 'shipped'),
(8,  16, 1, '2024-12-12', 'processing'),
(9,  11, 20,'2024-12-14', 'processing'),
(10, 4,  3, '2024-12-15', 'pending'),
(1,  5,  2, '2024-12-16', 'pending'),
(2,  8,  8, '2024-12-17', 'pending'),
(11, 6,  5, '2024-12-18', 'pending'),
(12, 9,  50,'2024-12-19', 'cancelled'),
(3,  13, 1, '2024-12-20', 'pending'),
(5,  14, 3, '2024-12-20', 'processing');
