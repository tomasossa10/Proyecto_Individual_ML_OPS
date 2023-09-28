from fastapi import FastAPI

app = FastAPI()

#http://127.0.0.1:8000

@app.get('/')
def index():
    return {'PI_ML_OPS' : 'Tomas Ossa'}

@app.get('/usersrecommend/{year}')
def UsersRecommend(year : int):
    import pandas as pd
    df = pd.read_csv('datasets/dataset_endpoint_1_2.csv')

    df_filtrado = df[df['year'] == year]

    df_ordenado = df_filtrado.sort_values(by='total_recomendaciones', ascending=False)

    top3_recomendaciones = df_ordenado.head(3)

    resultado = [{"Puesto {}: {}".format(i + 1, row['item_name']): row['total_recomendaciones']} for i, (_, row) in enumerate(top3_recomendaciones.iterrows())]

    return resultado

@app.get('/usersnotrecommend/{year}')
def UsersNotRecommend(year : int):
    import pandas as pd
    df = pd.read_csv('datasets/dataset_endpoint_1_2.csv')

    df_filtrado = df[df['year'] == year]

    df_ordenado = df_filtrado.sort_values(by='total_recomendaciones', ascending=True)

    top3_recomendaciones = df_ordenado.head(3)

    resultado = [{"Puesto {}: {}".format(i + 1, row['item_name']): row['total_recomendaciones']} for i, (_, row) in enumerate(top3_recomendaciones.iterrows())]

    return resultado

@app.get('/gamerecommend/{id}')
def Recomendaciones(id : int):
    import pandas as pd
    games = pd.read_csv('datasets/games_cleaned.csv')
    games_ml = pd.read_csv('datasets/dataset_modelo_ml.csv')
    games_ml.drop(columns=['publisher', 'app_name', 'developer', 'metascore'], inplace=True)
    
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False, drop='first')
    categorical_columns = ['genre1', 'genre2', 'genre3', 'genre4', 'genre5']
    encoded_categories = encoder.fit_transform(games_ml[categorical_columns])
    encoded_df = pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out(categorical_columns))
    games_encoded = pd.concat([games_ml, encoded_df], axis=1)
    games_encoded.drop(categorical_columns, axis=1, inplace=True)

    from sklearn.neighbors import NearestNeighbors
    n_neighbors=5
    nneighbors = NearestNeighbors(n_neighbors = n_neighbors, metric = 'cosine').fit(games_encoded)

    registro = games_encoded.loc[games['id'] == id].values.reshape(1, -1)
    distances, indices = nneighbors.kneighbors(registro)
    neighbor_data = games['app_name'].iloc[indices[0]]
    return neighbor_data