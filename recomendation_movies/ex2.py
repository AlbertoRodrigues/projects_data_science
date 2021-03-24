import numpy as np
import pandas as pd
from plotnine import *
pd.set_option("display.max_columns", 10)
rating=pd.read_csv("rating.csv")
anime=pd.read_csv("anime.csv")
rating.head()
anime.head()
#Colunas
anime.columns
#Quantidade de usuários
len(np.unique(rating["user_id"]))
#Quantidade de animes
len(np.unique(rating["anime_id"]))

#Tentando ver os gêneros únicos
ggplot(anime)+aes("genre")+geom_bar()
np.unique(anime["genre"],return_counts=True)
anime["genre"][1].split(",")

def formata_genero(x):
    return(x.split(","))
generos=anime["genre"].dropna().apply(formata_genero)
generos
type(generos[0])
generos.flatten("C")
np.ravel(generos)

anime["genre"][0].split(",")

generos.explode().value_counts()
generos.stack()
type(anime["episodes"][0])
pd.Series([1,2,3]).astype("str")
pd.__version__
#Analisando outras coisas
anime.head()
#def formata_episodes(x):
#    if x=="Unknown":
#        return(np.nan)
#    else:
#        return(int(x))
#anime["episodes2"]=anime["episodes"].apply(formata_episodes)

#pd.Series([np.nan,1,2]).dropna()
#anime["episodes"][12265]
#anime["type2"]=anime["episodes"].astype("int64")
np.unique(anime["type"].dropna(),return_counts=True)
ggplot(anime)+aes("type")+geom_bar(fill="blue")
#anime["episodes"][anime["episodes2"]>500]
#ggplot(anime)+aes("episodes2")+geom_histogram(fill="blue")+ylim([0,1600])
#anime["episodes2"].describe()
ggplot(anime)+aes("rating")+geom_histogram(fill="blue",color="black")

#Fazendo o merge e selecionando alguns animes por usuário.
rating_media=rating.groupby(['user_id']).mean().reset_index()
rating_p70=rating.groupby(['user_id']).quantile(q=0.70).reset_index()
rating_mediano=rating.groupby(['user_id']).median().reset_index()
rating_mediano['median_rating'] = rating_mediano['rating']
rating_mediano=rating_mediano.iloc[:,[0,3]]
rating_media['mean_rating'] = rating_media['rating']
rating_media=rating_media.iloc[:,[0,3]]
rating_p70['p70_rating'] = rating_p70['rating']
rating_p70=rating_p70.iloc[:,[0,3]]

rating_por_anime=pd.merge(rating,rating_media,on=['user_id','user_id'])

rating_geral=pd.merge(rating_mediano,rating_por_anime,on=['user_id','user_id'])
rating_geral.head()
rating_geral2=pd.merge(rating_geral,rating_p70,on=['user_id','user_id'])
#Excluir observações cujo rating é menor que a média ou a mediana?
#Melhor o percentil 70, pegando só os animes mais bem avaliados por pessoa
#Comparação entre os três gráficos
ggplot(rating_mediano)+aes("median_rating")+geom_histogram(fill="blue")
ggplot(rating_por_usuario)+aes("mean_rating")+geom_histogram(fill="blue")
ggplot(rating_p70)+aes("p70_rating")+geom_histogram(fill="blue")

animes_bem_avaliados=np.where(rating_geral2["rating"]>=rating_geral2["p70_rating"])[0]
len(animes_bem_avaliados)
dados_animes_bem_avaliados=rating_geral2.iloc[animes_bem_avaliados,:]
dados_animes_bem_avaliados.head()
pos=np.where(dados_animes_bem_avaliados["user_id"]<=10000)[0]
#pos=np.random.randint(0,len(np.unique(rating["user_id"])),12000)
#pos2=np.where(dados_animes_bem_avaliados["user_id"]==pos)[0]
dados_animes_bem_avaliados=dados_animes_bem_avaliados.iloc[pos,:]

matriz_ocorrencias=pd.crosstab(dados_animes_bem_avaliados["user_id"]
,dados_animes_bem_avaliados["anime_id"])
matriz_ocorrencias

np.sum(matriz_ocorrencias.values[:,1])

dados_animes_bem_avaliados.head()
m=dados_animes_bem_avaliados.pivot_table(index=['user_id'],
                            columns=['anime_id'], values='rating')
len(np.unique(dados_animes_bem_avaliados["anime_id"]))

m.replace({np.nan:0 }, regex=True, inplace = True)
m.replace({-1:0 }, regex=True, inplace = True)
m
from sklearn.cluster import KMeans
cluster=KMeans(n_clusters=3)
cluster.fit(m)
#predicao=cluster.predict(m)
#np.sum(cluster.labels_==predicao)
#cluster.inertia_
np.unique(cluster.labels_,return_counts=True)
matriz_ocorrencias["cluster"]=cluster.labels_
proporcao_anime0=np.mean(matriz_ocorrencias[matriz_ocorrencias["cluster"]==0].drop("cluster",axis=1),axis=0)
proporcao_anime1=np.mean(matriz_ocorrencias[matriz_ocorrencias["cluster"]==1].drop("cluster",axis=1),axis=0)
proporcao_anime2=np.mean(matriz_ocorrencias[matriz_ocorrencias["cluster"]==2].drop("cluster",axis=1),axis=0)

#Selecionando os animes mais vistos de cada cluster
proporcao_anime2.iloc[np.argsort(proporcao_anime2)[-10:-1]]

anime.head()
indices_cluster0=matriz_ocorrencias[matriz_ocorrencias["cluster"]==0].index
rating.head()