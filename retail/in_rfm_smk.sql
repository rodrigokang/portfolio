/* ****************************************************************** *
 * Título: Consulta RFM smk
 * ------------------------------------------------------------------ *
 * Analista: Jeremias K. Mazzetti
 * ------------------------------------------------------------------ *
 * Descripción: Esta consulta genera una tabla RFM (Recencia, Frecuen-
   cia, Monto).
 * ------------------------------------------------------------------ *
 * Argumentos:
 * - cadena = <Nombre de bandera>
 * - marca = <Nombre de la marca>
 * - region = <Nombre de región>
 * - locales = <Número de local>
 * - canal = <Tipo de canal>
 * - tipo_cliente = <Tipo de cliente>
 * - analista: <Nombre de la persona que ejecuta el modelo>
 * - ts_actualizacion: <Fecha en la que se realizó la ejecución>
 * - requerimiento: <Número de requerimiento>
 * - area_solicitante: <Sector que solicita el requerimiento>
 * - solicitante: <Responsable de la solicitud>
 * - date_start = <Fecha de inicio de la ventana de análisis>
 * - date_end = <Fecha de fin de la ventana de análisis>
 * - fecha_analisis = <Fecha de análisis>
 * - condicion_region = <Condición para la región>
 * - condicion_cadena = <Condición para la cadena>
 * - condicion_local = <Condición para el local>
 * - condicion_canal = <Condición para el canal>
 * - condicion_cliente = <Condición para el tipo de cliente>
 * - condicion_marca: <Condición para la marca del producto>
 * ****************************************************************** */
 
DROP TABLE IF EXISTS #pro;

SELECT 
    DISTINCT former_item_id
INTO #pro
FROM lk_cli_dim_vw.dim_item 
WHERE upper(brand_name) like upper('%{marca}%');
 

SELECT
    v.client_id AS idcliente,
    COUNT(DISTINCT v.reported_as_dt) AS frecuencia,
    SUM(v.ventaciva) AS monto,
    DATEDIFF(DAY, MAX(v.reported_as_dt), '{date_end}') AS recencia,
    CAST('{date_start}' AS date) AS fecha_inicio,
    CAST('{date_end}' AS date) AS fecha_fin,
    CAST('{fecha_analisis}' AS date) AS fecha_analisis,
    '{region}' AS region,
    '{cadena}' AS cadena,
    '{marca}' AS marca,
    '{nrolocal}' AS nrolocal,
    '{canal}' AS canal,
    '{tipo_cliente}' AS tipo_cliente,
    '{analista}' AS analista,
    '{ts_actualizacion}' AS ts_actualizacion,
    '{requerimiento}' AS requerimiento,
    '{area_solicitante}' AS area_solicitante,
    '{solicitante}' AS solicitante,
    DATEDIFF(DAY, '{date_start}', '{date_end}') AS ventana_tiempo
FROM
    lk_cli_fact_vw.fact_sales_transaction_detail v
JOIN 
    lk_analytics.Tsp_dim_locales l ON v.location_id = l.location_id
{condicion_cliente}
{condicion_marca}
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
    {condicion_canal}
GROUP BY
    v.client_id,
    fecha_analisis;