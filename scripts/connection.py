"""
    This short script serves as pipeline to the sql server we want to access, given the necessary params
"""
import pandas as pd
import numpy as np
import time
from sqlalchemy import create_engine


def get_db(params, args) -> pd.DataFrame:
    print('-'*30)
    print("Trying to connect to {0}:{1} with {2} as user".format(params["url_servidor"], params["puerto"], params["usuario"]))

    # Creamos una cadena de conexión válida que incluye todas las variables declaradas arriba.
    cadena_conexion = 'mysql+mysqlconnector://{0}:{1}@{2}:{3}/{4}?auth_plugin={5}'
    cadena_conexion = cadena_conexion.format(params["usuario"], params["contrasena"], params["url_servidor"], params["puerto"],
                                   params["esquema"], params["plugin_autenticacion"])
    conexion = create_engine(cadena_conexion)
        # Creamos consulta (podemos filtrarla para ir más rápido)
    if args.limit is not None:
        sql = 'SELECT * FROM data_input LIMIT {}'.format(args.limit)
    else:
        sql = 'SELECT * FROM data_input'

    df = pd.read_sql(sql, conexion)

    print("Connection established")
    print('-'*30)

    return df
