import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest,  mutual_info_classif
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from scipy.stats import uniform
from imblearn.under_sampling import NearMiss


#Descobrir quais variáveis 
X=pd.read_csv("train.csv")
y=X["TARGET"]
X_treino, X_teste, y_treino, y_teste=train_test_split(X, y, test_size=0.2, stratify=y, random_state=7)
X_treino=X_treino.drop(["TARGET","ID"],axis=1)
X_teste=X_teste.drop(["TARGET","ID"],axis=1)
remove = []
for col in X_treino.columns:
    if np.std(X_treino[col]) == 0:
        remove.append(col)
len(remove)

X_treino.drop(remove, axis=1, inplace=True)
X_teste.drop(remove, axis=1, inplace=True)
# remove duplicated columns
remove = []
cols = X_treino.columns
for i in range(len(cols)-1):
    v = X_treino[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,X_treino[cols[j]].values):
            remove.append(cols[j])
len(remove)
X_treino.drop(remove, axis=1, inplace=True)
X_teste.drop(remove, axis=1, inplace=True)
#Exportação para análise descritiva e criação de variáveis no R
X_treino.to_csv("x_treino.csv", index=False)
X_teste.to_csv("x_teste.csv", index=False)
y_treino.to_csv("y_treino.csv", index=False)
y_teste.to_csv("y_teste.csv", index=False)


modelo_adaboost1=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=7))
modelo_adaboost1.fit(X_treino,y_treino)
modelo_adaboost2=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=4))
modelo_adaboost2.fit(X_treino,y_treino)
modelo_adaboost3=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2))
modelo_adaboost3.fit(X_treino,y_treino)


feat_imp1 = pd.Series(modelo_adaboost1.feature_importances_, index = X_treino.columns.values).sort_values(ascending=False)
feat_imp1.iloc[:6]
feat_imp2 = pd.Series(modelo_adaboost2.feature_importances_, index = X_treino.columns.values).sort_values(ascending=False)
feat_imp2.iloc[:6]
feat_imp3 = pd.Series(modelo_adaboost3.feature_importances_, index = X_treino.columns.values).sort_values(ascending=False)
feat_imp3.iloc[:6]


#Importação de dados após a criação de variável possivelmente relevcantes
treino=pd.read_csv("treino_com_criacao_variaveis.csv")
teste=pd.read_csv("teste_com_criacao_variaveis.csv")
X=treino.drop(["TARGET"],axis=1)
y=treino["TARGET"]
np.unique(y,return_counts=True)
reamostragem=NearMiss(version=1, sampling_strategy={0:50000, 1:2406})
X, y = reamostragem.fit_resample(X, y)
#Seleção de variáveis

selector = SelectKBest(score_func=mutual_info_classif, k=61)
X_2=selector.fit_transform(X,y)
y_2=y.values

k_vs_roc_teste = []
k_vs_recall_0_teste = []
k_vs_recall_1_teste = []
k_vs_precision_0_teste = []
k_vs_precision_1_teste = []
k_vs_f1_1_teste = []
k_vs_f1_0_teste = []
k_vs_f1_teste = []

k_vs_roc_treino = []
k_vs_recall_0_treino = []
k_vs_recall_1_treino = []
k_vs_precision_0_treino = []
k_vs_precision_1_treino = []
k_vs_f1_1_treino = []
k_vs_f1_0_treino = []
k_vs_f1_treino = []
indice_i=[]

k=RepeatedStratifiedKFold(n_splits=3,n_repeats=1, random_state=7)
#warnings.filterwarnings("ignore")

for i in range(1,62,3):
    recall_0_teste=[]
    recall_1_teste=[]
    precision_0_teste=[]
    precision_1_teste=[]
    roc_auc_teste=[]
    f1_0_teste=[]
    f1_1_teste=[]
    f1_teste=[]
    
    recall_0_treino=[]
    recall_1_treino=[]
    precision_0_treino=[]
    precision_1_treino=[]
    roc_auc_treino=[]
    f1_0_treino=[]
    f1_1_treino=[]
    f1_treino=[]
    print(i)
    for treino_index, teste_index in k.split(X_2,y_2):       
        X_treino=X_2[treino_index,:] 
        X_teste=X_2[teste_index,:]
        y_treino= y_2[treino_index] 
        y_teste=y_2[teste_index]
        selector = SelectKBest(score_func=mutual_info_classif, k=i)
        X_treino2=selector.fit_transform(X_treino,y_treino)
        X_teste2=selector.transform(X_teste)
        print("Passou a selecao a variaveis")
           
        mdl = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4
                        ,class_weight={0:1,1:20}),n_estimators=150)
        mdl.fit(X_treino2, y_treino)
        
        y_prob_predict = mdl.predict_proba(X_teste2)[:,1]
        y_predict_teste = mdl.predict(X_teste2)
        y_predict_treino = mdl.predict(X_treino2)
        
        recall_0_teste.append(recall_score(y_teste, y_predict_teste, pos_label=0))
        recall_1_teste.append(recall_score(y_teste, y_predict_teste, pos_label=1))
        precision_0_teste.append(precision_score(y_teste, y_predict_teste, pos_label=0))
        precision_1_teste.append(precision_score(y_teste, y_predict_teste, pos_label=1))
        roc_auc_teste.append(roc_auc_score(y_teste, y_prob_predict))
        f1_0_teste.append(f1_score(y_teste, y_predict_teste, pos_label=0))
        f1_1_teste.append(f1_score(y_teste, y_predict_teste, pos_label=1))
        f1_teste.append(f1_score(y_teste, y_predict_teste, average="macro"))
        
        recall_0_treino.append(recall_score(y_treino, y_predict_treino, pos_label=0))
        recall_1_treino.append(recall_score(y_treino, y_predict_treino, pos_label=1))
        precision_0_treino.append(precision_score(y_treino, y_predict_treino, pos_label=0))
        precision_1_treino.append(precision_score(y_treino, y_predict_treino, pos_label=1))
        roc_auc_treino.append(roc_auc_score(y_treino, mdl.predict_proba(X_treino2)[:,1]))
        f1_0_treino.append(f1_score(y_treino, y_predict_treino, pos_label=0))
        f1_1_treino.append(f1_score(y_treino, y_predict_treino, pos_label=1))
        f1_treino.append(f1_score(y_treino, y_predict_treino, average="macro"))
        
    k_vs_roc_treino.append(np.mean(roc_auc_treino))
    k_vs_recall_0_treino.append(np.mean(recall_0_treino))
    k_vs_recall_1_treino.append(np.mean(recall_1_treino))
    k_vs_precision_0_treino.append(np.mean(precision_0_treino))
    k_vs_precision_1_treino.append(np.mean(precision_1_treino))
    k_vs_f1_0_treino.append(np.mean(f1_0_treino))
    k_vs_f1_1_treino.append(np.mean(f1_1_treino))
    k_vs_f1_treino.append(np.mean(f1_treino))
    
    k_vs_roc_teste.append(np.mean(roc_auc_teste))
    k_vs_recall_0_teste.append(np.mean(recall_0_teste))
    k_vs_recall_1_teste.append(np.mean(recall_1_teste))
    k_vs_precision_0_teste.append(np.mean(precision_0_teste))
    k_vs_precision_1_teste.append(np.mean(precision_1_teste))
    k_vs_f1_0_teste.append(np.mean(f1_0_teste))
    k_vs_f1_1_teste.append(np.mean(f1_1_teste))
    k_vs_f1_teste.append(np.mean(f1_teste))
       
    indice_i.append(i)

    print("Roc AUC treino: {} - Roc AUC teste: {}".format(np.mean(roc_auc_treino), np.mean(roc_auc_teste) ))
    print("Recall classe 0 treino: {} - Recall classe 0 teste: {}".format(np.mean(recall_0_treino), np.mean(recall_0_teste) ))
    print("Recall classe 1 treino: {} - Recall classe 1 teste: {}".format(np.mean(recall_1_treino), np.mean(recall_1_teste) ))
    print("Precision classe 0 treino: {} - Precision classe 0 teste: {}".format(np.mean(precision_0_treino), np.mean(precision_0_teste) ))
    print("Precision classe 1 treino: {} - Precision classe 1 teste: {}".format(np.mean(precision_1_treino), np.mean(precision_1_teste) ))
    print("F1 classe 0 treino: {} - F1 classe 0 teste: {}".format(np.mean(f1_0_treino), np.mean(f1_0_teste) ))
    print("F1 classe 1 treino: {} - F1 classe 1 teste: {}".format(np.mean(f1_1_treino), np.mean(f1_1_teste) ))
    print("F1 Geral treino: {} - F1 Geral teste: {}".format(np.mean(f1_treino), np.mean(f1_teste) ))
    print("\n")

metricas=pd.DataFrame({"indices":indice_i,"roc_auc_treino":k_vs_roc_treino,
"roc_auc_teste":k_vs_roc_teste, "recall_0_treino": k_vs_recall_0_treino,
"recall_0_teste":k_vs_recall_0_teste, "recall_1_treino": k_vs_recall_1_treino,
"recall_1_teste": k_vs_recall_1_teste, "precision_0_treino": k_vs_precision_0_treino,
"precision_0_teste": k_vs_precision_0_teste, "precision_1_treino": k_vs_precision_1_treino,
"precision_1_teste": k_vs_precision_1_teste, "f1_0_treino":k_vs_f1_0_treino,
"f1_0_teste": k_vs_f1_0_teste, "f1_1_treino": k_vs_f1_1_treino,
"f1_1_teste": k_vs_f1_1_teste, "f1_treino": k_vs_f1_treino,
"f1_teste": k_vs_f1_teste})

metricas.to_csv(path_or_buf="\\Users\\Alberto\\Desktop\\ime-usp\\aprendizagem_estatistica\\trabalho_aplicacao\\metricas.csv", index=False)

#Quantidade variáveis escolhidas pro enquanto: 31, escolhido pelas métricas
#de recall, precision e f1 Score
selector = SelectKBest(score_func=mutual_info_classif, k=31)
selector.fit(X,y)
#selector.get_support()
#Variáveis selecionadas
X.loc[:,selector.get_support()].columns
X_2=selector.transform(X)
y_2=y.values

#pd.DataFrame(X_2,columns=X.loc[:,selector.get_support()].columns)
#Modelagem preditiva
#,Análise discriminante linear,Análise discriminante quadrático
modelo1=LinearDiscriminantAnalysis()
modelo2=QuadraticDiscriminantAnalysis()

erro1,erro2=([],[])
cont=0
for treino_index, teste_index in k.split(X_2,y_2):
    cont=cont+1
    print(cont)
    xtreino=X_2[treino_index,:] 
    xteste=X_2[teste_index,:]
    ytreino= y_2[treino_index] 
    yteste=y_2[teste_index]
    
    modelo1.fit(xtreino,ytreino)
    modelo2.fit(xtreino,ytreino)
  
    f1_modelo1 = recall_score(yteste, modelo1.predict(xteste), pos_label=1)
    f1_modelo2 = recall_score(yteste, modelo2.predict(xteste), pos_label=1)

    erro1.append(f1_modelo1)
    erro2.append(f1_modelo2)
  
print(np.mean(erro1))
print(np.mean(erro2))
tabela_media=[np.mean(erro1),np.mean(erro2)]
desvio=[np.std(erro1),np.std(erro2)]
hiper_parametros=["Nenhum","Nenhum"]


#Floresta Aleatória
modelo=RandomForestClassifier(n_estimators=150)
hiperp = {"max_depth":[1,2,3,4,5],
"min_samples_split":[5,10,30,50],"min_samples_leaf":[5,10,15,25,30],
"class_weight":["balanced", {0:1,1:10},{0:1,1:20}]}
Otimizacao = RandomizedSearchCV(modelo, hiperp,cv=k, n_iter = 20
                    ,scoring="recall", random_state=0)
Otimizacao.fit(X_2, y_2)
Otimizacao.cv_results_
Otimizacao.best_params_
erro=Otimizacao.best_score_
tabela_media.append(erro)
ind_menor_erro=np.argsort(Otimizacao.cv_results_["mean_test_score"])[-1]
dp=Otimizacao.cv_results_["std_test_score"][ind_menor_erro]
desvio.append(dp)
hiper_parametros.append(Otimizacao.best_params_)
print("Random Forest finalizado!")


#AdaBoost
modelo=AdaBoostClassifier(n_estimators=200)
hiperp = { "base_estimator":[
    DecisionTreeClassifier(max_depth=3,class_weight={0:1,1:10}),
    DecisionTreeClassifier(max_depth=3,class_weight={0:1,1:20}),
    DecisionTreeClassifier(max_depth=3,class_weight="balanced"),
    DecisionTreeClassifier(max_depth=5,class_weight={0:1,1:10}),
    DecisionTreeClassifier(max_depth=5,class_weight={0:1,1:20}),
    DecisionTreeClassifier(max_depth=5,class_weight="balanced")]}
Otimizacao = GridSearchCV(modelo, hiperp,cv=k, scoring="recall")
Otimizacao.fit(X_2, y_2)
Otimizacao.cv_results_
Otimizacao.best_params_
erro=Otimizacao.best_score_
tabela_media.append(erro)
ind_menor_erro=np.argsort(Otimizacao.cv_results_["mean_test_score"])[-1]
dp=Otimizacao.cv_results_["std_test_score"][ind_menor_erro]
desvio.append(dp)
hiper_parametros.append(Otimizacao.best_params_)
print("AdaBoost finalizado!")

#GradientBoosting
modelo=GradientBoostingClassifier(n_estimators=200)
hiperp = { "max_depth":[3, 5, 7]}
Otimizacao = GridSearchCV(modelo, hiperp,cv=k, scoring="recall")
Otimizacao.fit(X_2, y_2)
Otimizacao.cv_results_
Otimizacao.best_params_
erro=Otimizacao.best_score_
tabela_media.append(erro)
ind_menor_erro=np.argsort(Otimizacao.cv_results_["mean_test_score"])[-1]
dp=Otimizacao.cv_results_["std_test_score"][ind_menor_erro]
desvio.append(dp)
hiper_parametros.append(Otimizacao.best_params_)
print("Gradient Boosting finalizado!")

#Regressão Logística
modelo=LogisticRegression(max_iter=500)
hiperp = { "class_weight":["balanced",{0:1,1:10}
,{0:1,1:20}], "C":uniform(loc=0,scale=5)}
Otimizacao = RandomizedSearchCV(modelo, hiperp,cv=k, n_iter = 20
                    ,scoring="recall", random_state=0)
Otimizacao.fit(X_2, y_2)
Otimizacao.cv_results_
Otimizacao.best_params_
erro=Otimizacao.best_score_
tabela_media.append(erro)
ind_menor_erro=np.argsort(Otimizacao.cv_results_["mean_test_score"])[-1]
dp=Otimizacao.cv_results_["std_test_score"][ind_menor_erro]
desvio.append(dp)
hiper_parametros.append(Otimizacao.best_params_)
print("Regressão Logística finalizada!")

tabela_media

hiper_parametros[2].update({"n_estimators":150})
hiper_parametros[3].update({"n_estimators":200})
hiper_parametros[4].update({"n_estimators":20})
hiper_parametros[5].update({"max_iter":500})

X_teste=teste.drop(["TARGET"], axis=1)
X_teste=selector.transform(X_teste)
y_teste=teste["TARGET"]

X_teste.shape

modelos=[LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(), RandomForestClassifier(**hiper_parametros[2]), AdaBoostClassifier(**hiper_parametros[3]), GradientBoostingClassifier(**hiper_parametros[4]), LogisticRegression(**hiper_parametros[5])]
#modelos=[LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(), RandomForestClassifier(**hiper_parametros[2]), AdaBoostClassifier(**hiper_parametros[3]), GradientBoostingClassifier(**hiper_parametros[4]), LogisticRegression(**hiper_parametros[5])]
for modelo in modelos:
    print("Modelo: ", modelo)
    modelo.fit(X_2,y_2)
    y_predict=modelo.predict(X_teste)
    
    print("Recall classe 0: ", recall_score(y_teste, y_predict, pos_label=0))
    print("Recall classe 1: ",recall_score(y_teste, y_predict, pos_label=1))
    print("Precision classe 0: ", precision_score(y_teste, y_predict, pos_label=0))
    print("Precision classe 1: ", precision_score(y_teste, y_predict, pos_label=1))
    print("F1 classe 0: ", f1_score(y_teste, y_predict, pos_label=0))
    print("F1 classe 1: ", f1_score(y_teste, y_predict, pos_label=1))
    print("F1 Score Médio: ", f1_score(y_teste, y_predict, average="macro"))
    print("\n")
    
##Vendo diferentes melhores para os modelos AdaBoost e GradientBoosting
modelo1=AdaBoostClassifier(n_estimators=200, base_estimator= DecisionTreeClassifier(class_weight={0: 1, 1: 10}, max_depth=3))
modelo2=GradientBoostingClassifier(n_estimators=200, max_depth= 7)

f1_classe0_modelo1=[]
f1_classe1_modelo1=[]
recall_classe0_modelo1=[]
recall_classe1_modelo1=[]
precision_classe0_modelo1=[]
precision_classe1_modelo1=[]
f1_geral_modelo1=[]
roc_auc_modelo1=[]

f1_classe0_modelo2=[]
f1_classe1_modelo2=[]
recall_classe0_modelo2=[]
recall_classe1_modelo2=[]
precision_classe0_modelo2=[]
precision_classe1_modelo2=[]
f1_geral_modelo2=[]
roc_auc_modelo2=[]
cont=0

for treino_index, teste_index in k.split(X_2,y_2):
    cont=cont+1
    
    xtreino=X_2[treino_index,:] 
    xteste=X_2[teste_index,:]
    ytreino= y_2[treino_index] 
    yteste=y_2[teste_index]
    
    modelo1.fit(xtreino,ytreino)
  
    prob_predict1 = modelo1.predict_proba(xteste)[:,1]
    y_predict1 = modelo1.predict(xteste)
    
    recall_classe0_modelo1.append(recall_score(yteste, y_predict1, pos_label=0))
    recall_classe1_modelo1.append(recall_score(yteste, y_predict1, pos_label=1))
    precision_classe0_modelo1.append(precision_score(yteste, y_predict1, pos_label=0))
    precision_classe1_modelo1.append(precision_score(yteste, y_predict1, pos_label=1))
    roc_auc_modelo1.append(roc_auc_score(yteste, prob_predict1))
    f1_classe0_modelo1.append(f1_score(yteste, y_predict1, pos_label=0))
    f1_classe1_modelo1.append(f1_score(yteste, y_predict1, pos_label=1))
    f1_geral_modelo1.append(f1_score(yteste, y_predict1, average="macro"))
        
    modelo2.fit(xtreino,ytreino)
    
    prob_predict2 = modelo2.predict_proba(xteste)[:,1]
    y_predict2 = modelo2.predict(xteste)
    
    recall_classe0_modelo2.append(recall_score(yteste, y_predict2, pos_label=0))
    recall_classe1_modelo2.append(recall_score(yteste, y_predict2, pos_label=1))
    precision_classe0_modelo2.append(precision_score(yteste, y_predict2, pos_label=0))
    precision_classe1_modelo2.append(precision_score(yteste, y_predict2, pos_label=1))
    roc_auc_modelo2.append(roc_auc_score(yteste, prob_predict2))
    f1_classe0_modelo2.append(f1_score(yteste, y_predict2, pos_label=0))
    f1_classe1_modelo2.append(f1_score(yteste, y_predict2, pos_label=1))
    f1_geral_modelo2.append(f1_score(yteste, y_predict2, average="macro"))
    print(cont)

np.mean(f1_classe0_modelo1)
np.mean(f1_classe1_modelo1)
np.mean(recall_classe0_modelo1)
np.mean(recall_classe1_modelo1)
np.mean(precision_classe0_modelo1)
np.mean(precision_classe1_modelo1)
np.mean(f1_geral_modelo1)
np.mean(roc_auc_modelo1)

np.mean(f1_classe0_modelo2)
np.mean(f1_classe1_modelo2)
np.mean(recall_classe0_modelo2)
np.mean(recall_classe1_modelo2)
np.mean(precision_classe0_modelo2)
np.mean(precision_classe1_modelo2)
np.mean(f1_geral_modelo2)
np.mean(roc_auc_modelo2)

metricas_boosting=pd.DataFrame({"roc_auc_modelo1":roc_auc_modelo1,
"roc_auc_modelo2":roc_auc_modelo2, "recall_0_modelo1": recall_classe0_modelo1,
"recall_0_modelo2":recall_classe0_modelo2, "recall_1_modelo1": recall_classe1_modelo1,
"recall_1_modelo2": recall_classe1_modelo2, "precision_0_modelo1": precision_classe0_modelo1,
"precision_0_modelo2": precision_classe0_modelo2, "precision_1_modelo1": precision_classe1_modelo1,
"precision_1_modelo2": precision_classe1_modelo2, "f1_0_modelo1":f1_classe0_modelo1,
"f1_0_modelo2": f1_classe0_modelo2, "f1_1_modelo1": f1_classe1_modelo1,
"f1_1_modelo2": f1_classe1_modelo2, "f1_modelo1": f1_geral_modelo1,
"f1_modelo2": f1_geral_modelo2})
metricas_boosting.to_csv(path_or_buf="\\Users\\Alberto\\Desktop\\ime-usp\\aprendizagem_estatistica\\trabalho_aplicacao\\metricas_boosting.csv", index=False)
#Stacking

cont=0
erro_stacking=[]
for treino_index, teste_index in k.split(X_2,y_2):
    cont=cont+1
    
    xtreino=X_2[treino_index,:] 
    xteste=X_2[teste_index,:]
    ytreino= y_2[treino_index] 
    yteste=y_2[teste_index]
    
    modelo1=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=7,class_weight={0:1,1:10}), n_estimators=200)
    modelo2=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=7,class_weight={0:1,1:20}), n_estimators=200)
    modelo3=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=7,class_weight="balanced"), n_estimators=200)
    modelo4=GradientBoostingClassifier(n_estimators=200, max_depth=3)
    modelo5=GradientBoostingClassifier(n_estimators=200, max_depth=5)
    modelo6=GradientBoostingClassifier(n_estimators=200, max_depth=7)
    modelo7=LogisticRegression()
    #modelo8=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5,class_weight={0:1,1:20}), n_estimators=200)
    #modelo9=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5,class_weight="balanced"), n_estimators=200)

    modelo1.fit(xtreino ,ytreino)
    modelo2.fit(xtreino ,ytreino)
    modelo3.fit(xtreino ,ytreino)
    modelo4.fit(xtreino ,ytreino)
    modelo5.fit(xtreino ,ytreino)
    modelo6.fit(xtreino ,ytreino)
    modelo7.fit(xtreino ,ytreino)
    #modelo8.fit(xtreino ,ytreino)
    #modelo9.fit(xtreino ,ytreino)
    
    probs1=modelo1.predict_proba(xtreino)
    probs2=modelo2.predict_proba(xtreino)
    probs3=modelo3.predict_proba(xtreino)
    probs4=modelo4.predict_proba(xtreino)
    probs5=modelo5.predict_proba(xtreino)
    probs6=modelo6.predict_proba(xtreino)
    probs7=modelo7.predict_proba(xtreino)
    #probs8=modelo8.predict_proba(xtreino)
    #probs9=modelo9.predict_proba(xtreino)
    x_stacking=np.column_stack((probs1[:,1], probs2[:,1], probs3[:,1], probs4[:,1], probs5[:,1], probs6[:,1], probs7[:,1]))
    modelo=LogisticRegression(class_weight= {0: 1, 1: 10})
    #, class_weight= {0: 1, 1: 10}
    modelo.fit(x_stacking, ytreino)
    
    probs1=modelo1.predict_proba(xteste)
    probs2=modelo2.predict_proba(xteste)
    probs3=modelo3.predict_proba(xteste)
    probs4=modelo4.predict_proba(xteste)
    probs5=modelo5.predict_proba(xteste)
    probs6=modelo6.predict_proba(xteste)
    probs7=modelo7.predict_proba(xteste)
    #probs8=modelo8.predict_proba(xteste)
    #probs9=modelo9.predict_proba(xteste)
    x_teste_stacking=np.column_stack((probs1[:,1], probs2[:,1], probs3[:,1], probs4[:,1], probs5[:,1], probs6[:,1], probs7[:,1]))

    f1_stacking = f1_score(yteste, modelo.predict(x_teste_stacking), pos_label=1)
    erro_stacking.append(f1_stacking)
    print(cont)

np.mean(erro_stacking)    
    
#Submissao
#Teste1: Melhor modelo AdaBoost com seleção de variáveis
teste=pd.read_csv("teste_com_criacao_variaveis.csv")
teste_ID=teste["ID"]
teste2=teste.drop("ID",axis=1)
teste2.shape
teste2=selector.transform(teste2)
modelo=AdaBoostClassifier(**Otimizacao.best_params_)
modelo.fit(X_2,y_2)
probs=modelo.predict_proba(teste2)
submission = pd.DataFrame({"ID":teste_ID, "TARGET": probs[:,1]})
submission.to_csv("submission_melhor_modelo_adaboost.csv", index=False)

#Teste2: Stacking AdaBoost 3 modelos específicos

hiperp = {"class_weight":["balanced",{0:1,1:10}
,{0:1,1:20}], "C":uniform(loc=0,scale=5)}
Otimizacao = RandomizedSearchCV(modelo, hiperp,cv=k, n_iter = 10
                    ,scoring="f1", random_state=0)
Otimizacao.fit(x_stacking, y_2)
Otimizacao.cv_results_
Otimizacao.best_params_
Otimizacao.best_score_
modelo=LogisticRegression(**Otimizacao.best_params_)


teste=pd.read_csv("teste_com_criacao_variaveis.csv")
teste_ID=teste["ID"]
teste2=teste.drop("ID",axis=1)
teste2=selector.transform(teste2)
probs1=modelo1.predict_proba(teste2)
probs2=modelo2.predict_proba(teste2)
probs3=modelo3.predict_proba(teste2)
probs4=modelo4.predict_proba(teste2)
probs5=modelo5.predict_proba(teste2)
probs6=modelo6.predict_proba(teste2)
#probs7=modelo7.predict_proba(teste2)
#probs8=modelo8.predict_proba(teste2)
#probs9=modelo9.predict_proba(teste2)
x_teste_stacking=np.column_stack((probs1[:,1], probs2[:,1], probs3[:,1], probs4[:,1], probs5[:,1], probs6[:,1]))
probs=modelo.predict_proba(x_teste_stacking)

submission = pd.DataFrame({"ID":teste_ID, "TARGET": probs[:,1]})
submission.to_csv("submission_stacking2.csv", index=False)

#Teste3: Gradient Boosting otimizado com NearMiss