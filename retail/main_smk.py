# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< main >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #

# Name: Implementación general de modelos smk
# Owner: Rodrigo J. Kang
# Descripción: Este script de python es el main para implementar los 
#              modelos basados en ML para smk en Cencosud S.A.

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

def main_rfm(date_start, date_end, cadena, marca, region, canal, nrolocal, tipo_cliente,
             analista, ts_actualizacion, requerimiento, area_solicitante, 
             solicitante, user, password, sql_script_file):
    """
    Ejecuta el proceso de segmentación RFM.

    Parámetros:
    - date_start (str): Fecha de inicio en formato 'YYYY-MM-DD'.
    - date_end (str): Fecha de fin en formato 'YYYY-MM-DD'.
    - cadena (str): Nombre de la cadena.
    - marca (str): Nombre de la marca.
    - region (str): Nombre de la región.
    - canal (str): Nombre del canal.
    - nrolocal (str): Número del local.
    - tipo_cliente (str): Tipo de cliente.
    - analista (str): Nombre de la persona que ejecuta el modelo.
    - ts_actualizacion (str): Fecha en la que se realizó la ejecución.
    - requerimiento (str): Número de requerimiento.
    - area_solicitante (str): Sector que solicita el requerimiento.
    - solicitante (str): Responsable de la solicitud.
    - user (str): Nombre de usuario para la conexión.
    - password (str): Contraseña para la conexión.
    - sql_script_file: Nombre del script .sql con la consulta input
    Devuelve:
    - output (DataFrame): DataFrame resultante de la segmentación RFM.
    """
    
    # Importar librería para leer archivos
    import os
    
    # Agregar la ruta del del módulo modelos
    import sys
    sys.path.append('C:/Users/RJKANG/Desktop/modelos_cencosud/')
    
    # Importar librería para manejar fechas
    from datetime import datetime, timedelta
    
    # Importar la clase RFM del módulo que contiene los modelos
    from modelos import RFM
    
    # Fecha de análisis date_end + 1
    fecha_analisis_dt = date_end + timedelta(days=1)
    fecha_analisis = fecha_analisis_dt.strftime('%Y-%m-%d')
    
    # ********************************************************************
    # Condiciones para parametrizar el script con la consulta
    # ********************************************************************
    
    # Argumentos region --------------------------------------------------
    if region == 'todas':
        condicion_region = """"""  
    else:
        condicion_region = f"""
                            AND LOWER(l.región) = LOWER('{region}')
                            """
    

    # Argumentos cadena --------------------------------------------------
    if cadena == 'smk':
        condicion_cadena = """"""
    elif cadena == 'jm':
        condicion_cadena = """
                           AND LOWER(l.cadena) in ('jumbo','disco')
                           """
    else:
        condicion_cadena = f"""
                            AND LOWER(l.cadena) = LOWER('{cadena}')
                            """
        
    # Argumentos marca -----------------------------------------
    if marca == 'todas':
        condicion_marca = """"""
    else:
        condicion_marca = """
                          INNER JOIN  #pro  p on v.former_item_id = p.former_item_id
                          """
    
    # Argumentos canal ----------------------------------------------------
    if canal == 'online':
        condicion_canal = """
                          AND v.sales_transaction_channel_cd in ('3', '4', '5', '6', '7', '8', '9')
                          """
    elif canal == 'presencial':
        condicion_canal = """
                          AND v.sales_transaction_channel_cd not in ('3', '4', '5', '6', '7', '8', '9')
                          """
    elif canal == 'omnicanal':
        condicion_canal = """"""

    # Argumentos local -----------------------------------------------------
    if nrolocal == 'todos':
        condicion_local = """"""  
    else:
        condicion_local = f"""
                           AND l.nrolocal = {nrolocal}
                           """
    
    # Argumentos tipo cliente ----------------------------------------------
    if tipo_cliente == 'prime':
        condicion_cliente = """
                            JOIN 
                                lk_mnc_vw.mncr_dclientes_dgrupoafinidad c 
                                ON c.idcliente = v.client_id 
                                AND idgrupoafinidad = 2046
                            """
    elif tipo_cliente == 'grandes socios':
        condicion_cliente = """
                            JOIN 
                                lk_mnc_vw.mncr_dclientes_dgrupoafinidad c 
                                ON c.idcliente = v.client_id 
                                AND idgrupoafinidad = 1816
                            """
    elif tipo_cliente == 'todos':
        condicion_cliente = """"""
    
    # ********************************************************************
    # Ejecutar el script .sql con los argumentos y las condiciones 
    # ********************************************************************

    # Ruta al archivo SQL
    sql_script_file = sql_script_file

    # Leer el contenido del archivo SQL y pasar los argumentos
    with open(sql_script_file, "r") as file:
        sql_script = file.read().format(
            date_start=date_start,
            date_end=date_end,
            fecha_analisis=fecha_analisis,
            region=region,
            cadena=cadena,
            marca=marca,
            nrolocal=nrolocal,
            canal=canal,
            analista=analista,
            ts_actualizacion=ts_actualizacion,
            requerimiento=requerimiento,
            area_solicitante=area_solicitante,
            solicitante=solicitante,
            tipo_cliente=tipo_cliente,
            condicion_region=condicion_region,
            condicion_cadena=condicion_cadena,
            condicion_local=condicion_local,
            condicion_canal=condicion_canal,
            condicion_cliente=condicion_cliente,
            condicion_marca=condicion_marca
        )

    # Crear una instancia de la clase RFM
    modelo_rfm = RFM(sql_script, user, password)

    # Procesar los datos
    modelo_rfm.preprocess_data()

    # Entrenar el modelo KMeans
    n_clusters = 8  # Se recomienda dejar el número de clusters fijo en 8
    modelo_rfm.train_kmeans(n_clusters)

    # DataFrame (Tabla Output) resultado
    output = modelo_rfm.segment_customers()
    
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #
    # Verificación de la segmentación
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

    #try:
    #    print("\n*************************")
    #    print("Información de los campos")
    #    print("*************************\n")

        # Imprimir los primeros registros del DataFrame resultado
    #    output.info()
    #except Exception as e:
    #    print()
    #    print(f"Ocurrió un error al imprimir la información de los campos: {e}")
    #    print("=========================================================")

    #try:
    #    print("\n*******************************")
    #    print("Verificar idclientes duplicados")
    #    print("*******************************\n")

        # Verificar si hay idcliente duplicados
    #    duplicates = output['idcliente'].duplicated().any()

    #    if duplicates:
    #        print("El DataFrame tiene idcliente duplicados.")
    #    else:
    #        print("El DataFrame no tiene idcliente duplicados.")
    #except Exception as e:
    #    print()
    #    print(f"Ocurrió un error al verificar idclientes duplicados: {e}")
    #    print("===================================================")

    #try:
    #    print("\n********************")
    #    print("Categorías distintas")
    #    print("********************\n")

        # Categorías distintas
    #    registros_distintos = output['categoria'].unique()

    #    print("\nCantidad de categorias únicas = ", len(registros_distintos))
    #    print("-----------------------------")
    #    for i in range(len(registros_distintos)):
    #        print(registros_distintos[i])
    #except Exception as e:
    #    print()
    #    print(f"Ocurrió un error al listar categorías distintas: {e}")
    #    print("===============================================")

    #try:
    #    print("\n******************************")
    #    print("Primeros registros de la tabla")
    #    print("******************************\n")

        # Imprimir los primeros registros del DataFrame resultado
    #    print(output.head())
    #except Exception as e:
    #    print()
    #    print(f"Ocurrió un error al imprimir los primeros registros de la tabla: {e}")
    #    print("===============================================================")
    
    return output

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

def main_churn_rfm(date_start_train, date_end_train, cadena, region, 
                   canal, nrolocal, tipo_cliente, analista, 
                   ts_actualizacion, user, password, sql_script_file,
                   churn_threshold, date_start_performance, 
                   date_end_performance):
    """
    Implementa los modelos de machine learning basados en RFM para predecir el churn en Cencosud S.A.

    Args:
        date_start_train (str): Fecha de inicio del período de entrenamiento en formato 'YYYY-MM-DD'.
        date_end_train (str): Fecha de fin del período de entrenamiento en formato 'YYYY-MM-DD'.
        cadena (str): Cadena de tiendas para el análisis.
        region (str): Región para el análisis.
        canal (str): Canal de ventas para el análisis.
        nrolocal (str): Número del local para el análisis.
        tipo_cliente (str): Tipo de cliente para el análisis.
        analista (str): Nombre del analista que ejecuta el script.
        ts_actualizacion (str): Fecha y hora de última actualización de los datos.
        user (str): Usuario para acceder a la base de datos.
        password (str): Contraseña para acceder a la base de datos.
        sql_script_file (str): Ruta al archivo SQL con la consulta necesaria para extraer los datos.
        churn_threshold (float): Umbral para clasificar churn.
        date_start_performance (str): Fecha de inicio del período de predicción en formato 'YYYY-MM-DD'.
        date_end_performance (str): Fecha de fin del período de predicción en formato 'YYYY-MM-DD'.

    Returns:
        DataFrame: Datos de salida con columnas de probabilidad y predicción añadidas.
    """
    import os
    import sys
    from datetime import datetime, timedelta
    from modelos import RFM, ChurnPredictorRfmLr
    import matplotlib.pyplot as plt

    sys.path.append('C:/Users/RJKANG/Desktop/modelos_cencosud/')

    # Fecha de análisis de entrenamiento date_end_train + 1
    date_end_train_dt = datetime.strptime(date_end_train, '%Y-%m-%d')
    fecha_analisis_train_dt = date_end_train_dt + timedelta(days=1)
    fecha_analisis_train = fecha_analisis_train_dt.strftime('%Y-%m-%d')
    
    # Fecha de análisis de predicción date_end_performance + 1
    date_end_performance_dt = datetime.strptime(date_end_performance, '%Y-%m-%d')
    fecha_analisis_performance_dt = date_end_performance_dt + timedelta(days=1)
    fecha_analisis = fecha_analisis_performance_dt.strftime('%Y-%m-%d')
    
    # Condiciones para parametrizar el script con la consulta
    
    # Argumentos region
    if region == 'todas':
        condicion_region = """"""
    else:
        condicion_region = f"""
                            AND LOWER(l.región) = LOWER('{region}')
                            """

    # Argumentos cadena
    if cadena == 'smk':
        condicion_cadena = """"""
    elif cadena == 'jm':
        condicion_cadena = """
                           AND LOWER(l.cadena) IN ('jumbo','disco')
                           """
    else:
        condicion_cadena = f"""
                            AND LOWER(l.cadena) = LOWER('{cadena}')
                            """
    
    # Argumentos canal
    if canal == 'online':
        condicion_canal = """
                          AND v.sales_transaction_channel_cd in ('3', '4', '5', '6', '7', '8', '9')
                          """
    elif canal == 'presencial':
        condicion_canal = """
                          AND v.sales_transaction_channel_cd not in ('3', '4', '5', '6', '7', '8', '9')
                          """
    elif canal == 'omnicanal':
        condicion_canal = """"""

    # Argumentos local
    if nrolocal == 'todos':
        condicion_local = """"""  
    else:
        condicion_local = f"""
                           AND l.nrolocal = {nrolocal}
                           """
    
    # Argumentos tipo cliente
    if tipo_cliente == 'prime':
        condicion_cliente = """
                            JOIN 
                                lk_mnc_vw.mncr_dclientes_dgrupoafinidad c 
                                ON c.idcliente = v.client_id 
                                AND idgrupoafinidad = 2046
                            """
    elif tipo_cliente == 'grandes socios':
        condicion_cliente = """
                            JOIN 
                                lk_mnc_vw.mncr_dclientes_dgrupoafinidad c 
                                ON c.idcliente = v.client_id 
                                AND idgrupoafinidad = 1816
                            """
    elif tipo_cliente == 'todos':
        condicion_cliente = ""

    # Ejecutar el script .sql para el entrenamiento
    with open(sql_script_file, "r") as file:
        sql_script_train = file.read().format(
            date_start=date_start_train,
            date_end=date_end_train,
            fecha_analisis=fecha_analisis_train,
            region=region,
            cadena=cadena,
            nrolocal=nrolocal,
            canal=canal,
            analista=analista,
            ts_actualizacion=ts_actualizacion,
            tipo_cliente=tipo_cliente,
            condicion_region=condicion_region,
            condicion_cadena=condicion_cadena,
            condicion_local=condicion_local,
            condicion_canal=condicion_canal,
            condicion_cliente=condicion_cliente,
        )

    modelo_train_churn = RFM(sql_script_train, user, password)
    modelo_train_churn.preprocess_data()
    n_clusters = 8  # Se recomienda dejar el número de clusters fijo en 8
    modelo_train_churn.train_kmeans(n_clusters)
    input_data = modelo_train_churn.segment_customers()

    churn_predictor = ChurnPredictorRfmLr()
    train_data = churn_predictor.preprocess_data(input_data)
    X_test, y_test = churn_predictor.train(train_data)
    
    print("\n****************************************************")
    y_pred, y_prob = churn_predictor.evaluate_model(X_test, y_test)
    print("****************************************************\n")

    # Graficar la curva ROC
    #churn_predictor.plot_roc_curve(y_test, y_prob)

    # Graficar la curva KS
    #churn_predictor.plot_ks_curve(y_test, y_prob)

    # Ejecutar el script .sql para la predicción
    with open(sql_script_file, "r") as file:
        sql_script_performance = file.read().format(
            date_start=date_start_performance,
            date_end=date_end_performance,
            fecha_analisis=fecha_analisis,
            region=region,
            cadena=cadena,
            nrolocal=nrolocal,
            canal=canal,
            analista=analista,
            ts_actualizacion=ts_actualizacion,
            tipo_cliente=tipo_cliente,
            condicion_region=condicion_region,
            condicion_cadena=condicion_cadena,
            condicion_local=condicion_local,
            condicion_canal=condicion_canal,
            condicion_cliente=condicion_cliente,
        )
    
    modelo_performance_churn = RFM(sql_script_performance, user, password)
    modelo_performance_churn.preprocess_data()
    modelo_performance_churn.train_kmeans(n_clusters)
    output_churn = modelo_performance_churn.segment_customers()
    
    performance_data = churn_predictor.preprocess_data(output_churn)
    output = churn_predictor.predict(performance_data, 
                                     churn_threshold=churn_threshold)
    
    return output

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

def get_parameters(user, password):
    """
    Obtiene los parámetros desde la base de datos y los devuelve 
    como un DataFrame.

    Args:
    user (str): Usuario de la base de datos.
    password (str): Contraseña del usuario de la base de datos.

    Returns:
    pandas.DataFrame: DataFrame que contiene todos los parámetros 
    obtenidos de la base de datos.
    """
    
    # Importar librerías
    import pandas as pd
    import psycopg2
    import warnings

    # Suprimir todas las advertencias
    warnings.filterwarnings("ignore")
    
    try:
        conn = psycopg2.connect(
            dbname='arsuperlake_prod',
            user=user,
            password=password,
            host='datalake-dw55-prod.ctnjyflrnhjv.us-east-1.redshift.amazonaws.com',
            port='5439'
        )
        query = """
                SELECT 
                    * 
                FROM 
                    lk_analytics.rfm_smk_requerimientos;
                """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error al conectar con la base de datos: {e}")
        return pd.DataFrame()

def check_records(table_name, requerimiento, user, password):
    """
    Verifica si hay registros en la base de datos para un requerimiento dado.

    Args:
    table_name (str): Nombre de la tabla donde verificar los registros.
    requerimiento (int): El número de requerimiento.
    user (str): Usuario de la base de datos.
    password (str): Contraseña del usuario de la base de datos.

    Returns:
    bool: True si hay registros, False en caso contrario.
    """
    
    # Importar librerías
    import psycopg2
    import warnings

    # Suprimir todas las advertencias
    warnings.filterwarnings("ignore")
    
    try:
        # Conexión a la base de datos
        conn = psycopg2.connect(
            dbname='arsuperlake_prod',
            user=user,
            password=password,
            host='datalake-dw55-prod.ctnjyflrnhjv.us-east-1.redshift.amazonaws.com',
            port='5439'
        )
        
        # Verificar si el requerimiento existe en la tabla especificada
        query_requerimiento = f"""
                              SELECT 
                                  COUNT(*) 
                              FROM 
                                  lk_analytics.{table_name} 
                              WHERE 
                                  requerimiento = '{requerimiento}';
                              """
        cur = conn.cursor()
        cur.execute(query_requerimiento)
        requerimiento_count = cur.fetchone()[0]
        
        if requerimiento_count > 0:
            print("\n------------------------------------------------------------")
            print(f"Requerimiento {requerimiento} ya existe en {table_name} -> no se procesa")
            print("------------------------------------------------------------")
            # Cerrar la conexión y devolver True para indicar que ya existe
            cur.close()
            conn.close()
            return True
        
        print("\n----------------------------------------------------------------")
        print(f"Requerimiento {requerimiento} no encontrado en {table_name} -> se procesa")
        print("----------------------------------------------------------------")
        
        # Cerrar la conexión
        cur.close()
        conn.close()
        
        return False
    except Exception as e:
        print(f"Error al verificar registros: {e}")
        return False

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #