-- last epoch + stats
SELECT
    e.experiment_id,
    e.cx,
    e.fun,
    (JULIANDAY(time.eval_end) - JULIANDAY(time.eval_start)) * 86400.0 AS evaluation_time,
    time.eval_start,
    time.eval_end,
    e.dim,
    e.min,
    e.max,
    e.avg,
    e.std,
    e.config
FROM experiments AS e join (
    SELECT experiment_id,
        MIN(date) AS eval_start,
        MAX(date) AS eval_end,
        MAX(id) AS last_id
    FROM experiments
        GROUP BY experiment_id) as time
    on time.experiment_id = e.experiment_id and time.last_id = e.id;