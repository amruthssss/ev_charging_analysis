-- Average charging duration (minutes) per station
SELECT charging_station_id,
       AVG(charging_duration_minutes) AS avg_duration_minutes
FROM ev_sessions
GROUP BY charging_station_id
ORDER BY avg_duration_minutes DESC;
