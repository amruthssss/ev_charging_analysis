-- Station demand: total sessions per charging station
SELECT charging_station_id,
       COUNT(*) AS sessions
FROM ev_sessions
GROUP BY charging_station_id
ORDER BY sessions DESC;
