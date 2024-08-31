/* ****************************************************************** *
* Título: Consulta Abandonadores
* ------------------------------------------------------------------- *
* Analista: Jeremias K. Mazzetti
* ------------------------------------------------------------------- *
* Descripción: Esta consulta genera una tabla RFM (Recencia, Frecuen-
   cia, Monto) que hace de input del modelo de Abandonadores.
* ------------------------------------------------------------------- *
* Argumentos:
* - cadena = <Nombre de bandera>
* - region = <Nombre de región>
* - locales = <Número de local>
* - canal = <Tipo de canal>
* - tipo_cliente = <Tipo de cliente>
* - date_start = <Fecha de inicio de la ventana de análisis>
* - date_end = <Fecha de fin de la ventana de análisis>
* - fecha_analisis = <Fecha de análisis>
* - condicion_region = <Condición para la región>
* - condicion_cadena = <Condición para la cadena>
* - condicion_local = <Condición para el local>
* - condicion_canal = <Condición para el canal>
* - condicion_cliente = <Condición para el tipo de cliente>
* ******************************************************************* */
 
DROP TABLE IF EXISTS #clientes_de_interes;
SELECT DISTINCT
    client_id idcliente,
    CAST(reported_as_dt AS DATE) AS reported_as_dt
INTO #clientes_de_interes
FROM
    lk_cli_fact_vw.fact_sales_transaction v
JOIN
    lk_analytics.Tsp_dim_locales l ON v.location_id = l.location_id
    {condicion_cliente}
WHERE
    v.sales_doc_type_cd in ('01', '02', '03', '04', '05', '06', '07',
                            '08', '09', '11', '12', '13', '14', '15',
                            '16', '17', '18', '19', 'T')
    AND v.tran_type_cd = '00'
    AND v.reported_as_dt BETWEEN '{date_start}' AND '{date_end}'
    AND v.ventaciva > 0
    AND v.client_id > 0
    {condicion_region}
    {condicion_cadena}
    {condicion_local}
    {condicion_canal};
 
DROP TABLE IF EXISTS #dias_con_compras;
SELECT DISTINCT
    idcliente,
    reported_as_dt
INTO #dias_con_compras
FROM #clientes_de_interes;
 
DROP TABLE IF EXISTS #cliente_pares_compras;
SELECT
    idcliente,
    reported_as_dt,
    LAG(reported_as_dt) OVER (PARTITION BY idcliente ORDER BY reported_as_dt) AS prev_reported_as_dt
INTO #cliente_pares_compras
FROM #dias_con_compras;
 
DROP TABLE IF EXISTS #clientes_con_diferencia;
SELECT
    idcliente,
    reported_as_dt,
    prev_reported_as_dt,
    DATEDIFF(day, prev_reported_as_dt, reported_as_dt) AS diff_in_days
INTO #clientes_con_diferencia
FROM #cliente_pares_compras
WHERE prev_reported_as_dt IS NOT NULL;
 
DROP TABLE IF EXISTS #lantencia_promedio_x_cliente;
SELECT
    idcliente,
    AVG(diff_in_days) AS avg_latency_days
INTO #lantencia_promedio_x_cliente
FROM #clientes_con_diferencia
GROUP BY idcliente;
 
SELECT
    la.idcliente,
    COUNT(DISTINCT v.sales_tran_id) AS frecuencia,
    SUM(v.ventaciva) AS monto,
    DATEDIFF(DAY, MAX(v.reported_as_dt), '{date_end}') AS recencia,
    la.avg_latency_days AS latencia_promedio,
    CAST('{date_start}' AS DATE) AS fecha_inicio,
    CAST('{date_end}' AS DATE) AS fecha_fin,
    CAST('{fecha_analisis}' AS DATE) AS fecha_analisis,
    '{region}' AS region,
    '{cadena}' AS cadena,
    '{nrolocal}' AS nrolocal,
    '{canal}' AS canal,
    '{tipo_cliente}' AS tipo_cliente,
    '{analista}' AS analista,
    '{ts_actualizacion}' AS ts_actualizacion,
    DATEDIFF(DAY, '{date_start}', '{date_end}') AS ventana_tiempo
FROM lk_cli_fact_vw.fact_sales_transaction v
JOIN lk_analytics.Tsp_dim_locales l ON v.location_id = l.location_id
JOIN #lantencia_promedio_x_cliente la ON v.client_id = la.idcliente
WHERE
    v.sales_doc_type_cd IN ('01', '02', '03', '04', '05', '06', '07', '08', '09', '11', '12', '13', '14', '15', '16', '17', '18', '19', 'T')
    AND v.tran_type_cd = '00'
    AND v.reported_as_dt BETWEEN '{date_start}' AND '{date_end}'
    AND v.ventaciva > 0
    AND v.client_id > 0
    {condicion_region}
    {condicion_cadena}
    {condicion_local}
    {condicion_canal}
GROUP BY
    la.idcliente,
    la.avg_latency_days,
    fecha_analisis;