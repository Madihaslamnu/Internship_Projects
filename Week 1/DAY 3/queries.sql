-- 1. Top 10 product categories by total revenue
--Categories like beleza_saude, relogios_presentes, and 
--cama_mesa_banho (these top 3) generate the highest overall revenue 
--for the platform. 
SELECT p.product_category_name, SUM(o.price) AS total_revenue
FROM order_items o
INNER JOIN products p ON o.product_id = p.product_id
GROUP BY p.product_category_name
ORDER BY total_revenue DESC
LIMIT 10;

-- 2. Repeat customer rate: customers with more than 1 order
--This query calculates the number of unique customers which is around 2997
-- who have placed more than one order,indicating a repeat customer rate.
SELECT COUNT(*) AS repeat_customers
FROM (
    SELECT c.customer_unique_id, COUNT(o.order_id) AS order_count
    FROM orders o
    INNER JOIN customers c ON o.customer_id = c.customer_id
    GROUP BY c.customer_unique_id
    HAVING COUNT(o.order_id) > 1
) AS repeat_customers_table;

-- 3. Average delivery time by state
--This query calculates the average delivery time in days for each state,
-- providing insights into regional delivery performance.
SELECT c.customer_state,
       AVG(EXTRACT(EPOCH FROM (o.order_delivered_customer_date::timestamp - o.order_purchase_timestamp::timestamp)) / 86400) AS avg_delivery_days
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_delivered_customer_date IS NOT NULL
GROUP BY c.customer_state
ORDER BY avg_delivery_days;

-- 4. Rank sellers by total revenue within each state using window functions
--In this query, we calculate the total revenue for each seller and rank them 
--within their respective states based on their revenue.
WITH seller_revenue AS (
    SELECT s.seller_id, s.seller_state, SUM(o.price) AS total_revenue
    FROM sellers s
    JOIN order_items o ON s.seller_id = o.seller_id
    GROUP BY s.seller_id, s.seller_state
)
SELECT seller_id, seller_state, total_revenue,
       RANK() OVER (PARTITION BY seller_state ORDER BY total_revenue DESC) AS revenue_rank
FROM seller_revenue;

-- 5. Month-over-month order volume change per category
--This query calculates the order volume for each product category on a monthly basis
--and determines the change in volume from the previous month.
WITH monthly_category_orders AS (
    SELECT p.product_category_name,
           DATE_TRUNC('month', o.order_purchase_timestamp::timestamp) AS order_month,
           COUNT(DISTINCT o.order_id) AS order_volume
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE p.product_category_name IS NOT NULL
    GROUP BY p.product_category_name, order_month
)
SELECT product_category_name, order_month, order_volume,
       LAG(order_volume, 1) OVER (PARTITION BY product_category_name ORDER BY order_month) AS prev_month_volume,
       order_volume - LAG(order_volume, 1) OVER (PARTITION BY product_category_name ORDER BY order_month) AS volume_change
FROM monthly_category_orders;

-- 6. Cumulative running total revenue by month
--This query calculates the total revenue generated each month and provides 
--a cumulative running total of revenue over time.
WITH monthly_revenue AS (
    SELECT DATE_TRUNC('month', o.order_purchase_timestamp::timestamp) AS order_month,
           SUM(oi.price) AS monthly_revenue
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY order_month
)
SELECT order_month, monthly_revenue,
       SUM(monthly_revenue) OVER (ORDER BY order_month) AS cumulative_revenue
FROM monthly_revenue;

-- 7. Identify customer signup cohorts based on their first purchase month
--This query identifies the cohort of customers based on the month of their first purchase,
-- allowing for analysis of customer acquisition trends over time.
WITH customer_cohorts AS (
    SELECT c.customer_unique_id,
           MIN(DATE_TRUNC('month', o.order_purchase_timestamp::timestamp)) AS cohort_month
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    GROUP BY c.customer_unique_id
)
SELECT cohort_month, COUNT(customer_unique_id) AS total_new_customers
FROM customer_cohorts
GROUP BY cohort_month
ORDER BY cohort_month;

-- 8. Map individual customer active months for retention analysis
--This query maps each unique customer to the months in which they made purchases,
-- providing a basis for retention analysis and understanding customer engagement over time.
SELECT DISTINCT c.customer_unique_id,
       DATE_TRUNC('month', o.order_purchase_timestamp::timestamp) AS order_month
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id;

-- 9. Multi-step cohort retention tracking repeat purchases across subsequent months
--This query combines the cohort identification and customer activity mapping to analyze retention,
-- showing how many customers from each cohort made purchases in subsequent months.
WITH customer_cohorts AS (
    SELECT c.customer_unique_id,
           MIN(DATE_TRUNC('month', o.order_purchase_timestamp::timestamp)) AS cohort_month
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    GROUP BY c.customer_unique_id
),
customer_activity AS (
    SELECT DISTINCT c.customer_unique_id,
           DATE_TRUNC('month', o.order_purchase_timestamp::timestamp) AS order_month
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
)
SELECT co.cohort_month,
       ca.order_month,
       COUNT(DISTINCT ca.customer_unique_id) AS active_customers
FROM customer_cohorts co
JOIN customer_activity ca ON co.customer_unique_id = ca.customer_unique_id
GROUP BY co.cohort_month, ca.order_month
ORDER BY co.cohort_month, ca.order_month;

-- 10. Average order value breakdown by payment type
--This query calculates the average order value for each payment type, providing insights into 
--customer spending behavior based on their preferred payment methods.
SELECT p.payment_type, 
       COUNT(DISTINCT p.order_id) AS total_orders,
       AVG(p.payment_value) AS avg_payment_value
FROM payments p
GROUP BY p.payment_type
ORDER BY avg_payment_value DESC;