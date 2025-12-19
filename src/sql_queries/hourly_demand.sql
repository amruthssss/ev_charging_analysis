-- Hourly demand: session counts per start_hour
SELECT start_hour,
       COUNT(*) AS total_sessions
FROM ev_sessions
GROUP BY start_hour
ORDER BY start_hour;
