-- Monthly session trends using charging_start_time
SELECT strftime('%Y', charging_start_time) AS year,
       strftime('%m', charging_start_time) AS month,
       COUNT(*) AS sessions
FROM ev_sessions
GROUP BY year, month
ORDER BY year, month;
