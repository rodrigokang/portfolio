/* ****************************************************************** *
 * Título: Consulta Reglas de Asociación CM
 * ------------------------------------------------------------------ *
 * Analista: Martín A. Nogueroles
 * ------------------------------------------------------------------ *
 * Descripción: Esta consulta genera una tabla de transacciones por 
                dw_ticket.
 * ------------------------------------------------------------------ *
 * Argumentos:
 * - date_start = <Fecha de inicio de la ventana de análisis>
 * - date_end = <Fecha de fin de la ventana de análisis>
 * ****************************************************************** */

SELECT 
    *
FROM
    lk_analytics.sub_35_jumbo
WHERE
    fecha >= '{date_start}' 
    AND fecha < '{date_end}'