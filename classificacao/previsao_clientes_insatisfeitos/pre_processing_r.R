require(ggplot2)
require(gridExtra)
X_treino=read.csv(file.choose())
X_teste=read.csv(file.choose())
y_treino=read.csv(file.choose())
y_teste=read.csv(file.choose())

table(dados[amostra,"TARGET"])

X=cbind.data.frame(X_treino,y_treino)
teste=cbind.data.frame(X_teste, y_teste)

table(X$TARGET)/nrow(X)
summary(dados)

a=(ggplot(X)+aes(factor(TARGET),y = prop.table(stat(count)), 
           fill = factor(TARGET), label = scales::percent(prop.table(stat(count))))
   +geom_bar()
   +scale_fill_manual("Clientes", values = c("#E41A1C", "#377EB8"),
       labels=c("Satisfeitos", "Insatisfeitos"))
   +geom_text(stat = 'count',
              position = position_dodge(.9), 
              vjust = -0.5, 
              size = 3) + 
      scale_y_continuous(labels = scales::percent)
   +labs(x = ' ', y = 'Proporção',fill="Quantidade",title="Antes do NearMiss")
   +theme_classic())
NearMiss=data.frame("classe"=c(rep(0,50000),rep(1,2406)))
b=(ggplot(NearMiss)+aes(factor(classe),y = prop.table(stat(count)), 
               fill = factor(classe), label = scales::percent(prop.table(stat(count))))
   +geom_bar()
   +scale_fill_manual("Clientes", values = c("#E41A1C", "#377EB8"),
                      labels=c("Satisfeitos", "Insatisfeitos"))
   +geom_text(stat = 'count',
              position = position_dodge(.9), 
              vjust = -0.5, 
              size = 3) + 
      scale_y_continuous(labels = scales::percent)
   +labs(x = ' ', y = 'Proporção',fill="Quantidade",title="Depois do NearMiss")
   +theme_classic())
grid.arrange(a,b,ncol=2)

table(dados$TARGET)
table(NearMiss$classe)
xtable(rbind.data.frame(table(dados$TARGET), table(NearMiss$classe)))

#Aparenemene nenhum valor faltante
#nrow(dados[!complete.cases(dados),])/nrow(dados)
apply(dados,2,function(x) sum(is.na(x)))
sum(apply(dados,2,function(x) var(x))==0)


#Criação de variável, número de zeros de uma observação
X$quant_zero=apply(X,1,function(x) sum(x==0))
teste$quant_zero=apply(teste,1,function(x) sum(x==0))

a=(ggplot(X[X$TARGET==0,])+aes(quant_zero)+
      geom_histogram(aes(y=..density..),color="black",fill="#00AFBB")
   +theme_classic()
+labs(x="Quantidade de zeros", title="Clientes Satisfeitos"))
b=(ggplot(X[X$TARGET==1,])+aes(quant_zero)+
      geom_histogram(aes(y=..density..),color="black",fill="#00AFBB")
   +theme_classic()
   +labs(x="Quantidade de zeros", title="Clientes Insatisfeitos"))
grid.arrange(a,b,ncol=2)
summary(X$quant_zero[X$TARGET==0])
summary(X$quant_zero[X$TARGET==1])

#Variável número de produtos
X$num_var4

#TROCAR dados por X!!!!!!

#ggplot(dados[dados$TARGET==1,])+aes(num_var4)+geom_bar(color="black",fill="#00AFBB")+theme_classic()
a=(ggplot(X[X$TARGET==0,])+aes(factor(num_var4),y = prop.table(stat(count)), 
                                       fill = factor(num_var4), label = scales::percent(prop.table(stat(count))))
   +geom_bar()
   +scale_fill_brewer(palette = "Set1")
   +geom_text(stat = 'count',
              position = position_dodge(.9), 
              vjust = -0.5, 
              size = 3) + 
     scale_y_continuous(labels = scales::percent)
   +labs(x = 'Quantidade de produtos', y = 'Proporção',fill="Quantidade",title="Clientes Satisfeitos")
   +theme_classic())

b=(ggplot(X[X$TARGET==1,])+aes(factor(num_var4),y = prop.table(stat(count)), 
   fill = factor(num_var4), label = scales::percent(prop.table(stat(count))))
  +geom_bar()
  +scale_fill_brewer(palette = "Set1")
  +geom_text(stat = 'count',
             position = position_dodge(.9), 
             vjust = -0.5, 
             size = 3) + 
    scale_y_continuous(labels = scales::percent)
  +labs(x = 'Quantidade de produtos', y = 'Proporção',fill="Quantidade"
        ,title="Clientes Insatisfeitos")
  +theme_classic())


grid.arrange(a,b,ncol=2)
#Criação de variável qualitativa
X$quant_produto0=ifelse(X$num_var4==0,1,0)
X$quant_produto1=ifelse(X$num_var4==1,1,0)
teste$quant_produto0=ifelse(teste$num_var4==0,1,0)
teste$quant_produto1=ifelse(teste$num_var4==1,1,0)


#Variável var38
dados$var38

ggplot(X)+aes(var38)+geom_histogram(color="black",fill="#00AFBB")+theme_classic()

(ggplot(X[X$var38!=117310.979016494,])
+aes(var38)+geom_histogram(color="black",fill="#00AFBB")+theme_classic())

sort(table(X$var38),decreasing = T)[1:5]
ggplot(X)+aes(log(var38))+geom_histogram(color="black",fill="#00AFBB")+theme_classic()
#Moda
sort(table(X$var38),decreasing = T)[1]
mean(X$var38[dados$var38!=117310.979016494])

(ggplot(X[X$var38!=117310.979016494,])
+aes(log(var38))+geom_histogram(color="black",fill="#00AFBB")+theme_classic())

ggplot(X)+aes(log(var38))+geom_histogram(aes(y=..density..),color="black",fill="#00AFBB")+theme_classic()+facet_grid(~factor(TARGET))
ggplot(X[X$var38!=117310.979016494,])+aes(log(var38))+geom_histogram(aes(y=..density..),color="black",fill="#00AFBB")+theme_classic()+facet_grid(~factor(TARGET))

X$var38_normal=ifelse(X$var38!=117310.979016494,log(X$var38),0)
X$var38_normal_dummy=ifelse(X$var38!=117310.979016494,1,0)
teste$var38_normal=ifelse(teste$var38!=117310.979016494,log(teste$var38),0)
teste$var38_normal_dummy=ifelse(teste$var38!=117310.979016494,1,0)

#Variável criada 1
#(ggplot(X)+aes(var38_normal)+geom_histogram(color="black",fill="#00AFBB")+theme_classic())
ggplot(X)+aes(var38_normal)+geom_histogram(aes(y=..density..),color="black",fill="#00AFBB")+theme_classic()+facet_grid(~factor(TARGET))

#Variável criada 2
a=(ggplot(X[X$TARGET==0,])+aes(factor(var38_normal_dummy),y = prop.table(stat(count)), 
                                       fill = factor(var38_normal_dummy), label = scales::percent(prop.table(stat(count))))
   +geom_bar()
   +scale_fill_brewer(palette = "Set1")
   +geom_text(stat = 'count',
              position = position_dodge(.9), 
              vjust = -0.5, 
              size = 3) + 
     scale_y_continuous(labels = scales::percent)
   +labs(x = ' ', y = 'Proporção',fill="Quantidade",title="Clientes Satisfeitos")
   +theme_classic());a

b=(ggplot(X[X$TARGET==1,])+aes(factor(var38_normal_dummy),y = prop.table(stat(count)), 
                               fill = factor(var38_normal_dummy), label = scales::percent(prop.table(stat(count))))
   +geom_bar()
   +scale_fill_brewer(palette = "Set1")
   +geom_text(stat = 'count',
              position = position_dodge(.9), 
              vjust = -0.5, 
              size = 3) + 
     scale_y_continuous(labels = scales::percent)
   +labs(x = ' ', y = 'Proporção',fill="Quantidade",title="Clientes Insatisfeitos")
   +theme_classic());b
grid.arrange(a,b,ncol=2)

#Variável 15
X$var15
ggplot(X)+aes(var15)+geom_histogram(color="black",fill="#00AFBB")+theme_classic()
a=(ggplot(X[X$TARGET==0,])+aes(var15)+geom_histogram(aes(y=..density..),color="black",fill="#00AFBB")+theme_classic()
+labs(x="Idade",title="Clientes Satisfeitos"))
b=(ggplot(X[X$TARGET==1,])+aes(var15)+geom_histogram(aes(y=..density..),color="black",fill="#00AFBB")+theme_classic()
   +labs(x="Idade",title="Clientes Insatisfeitos"))
grid.arrange(a,b,ncol=2)
ggplot(X[X$TARGET==1,])+aes(var15)+geom_histogram(color="black",fill="#00AFBB")+theme_classic()

summary(X$var15[X$TARGET==0])
summary(X$var15[X$TARGET==1])
table(X$var15[X$TARGET==0])[1:20]
table(X$var15[X$TARGET==1])[1:5]


X$idade_menor=ifelse(X$var15<=21,1,0)
teste$idade_menor=ifelse(teste$var15<=21,1,0)
#Variável 36
X$var36
table(X$var36[X$TARGET==0])
table(X$var36[X$TARGET==1])
a=(ggplot(X[X$TARGET==0,])+aes(factor(var36),y = prop.table(stat(count)), 
                                       fill = factor(var36), label = scales::percent(prop.table(stat(count))))
   +geom_bar()
   +scale_fill_brewer(palette = "Set1")
   +geom_text(stat = 'count',
              position = position_dodge(.9), 
              vjust = -0.5, 
              size = 3) + 
     scale_y_continuous(labels = scales::percent)
   +labs(x = ' ', y = 'Proporção',fill="Quantidade",title="Clientes Satisfeitos")
   +theme_classic())

b=(ggplot(X[X$TARGET==1,])+aes(factor(var36),y = prop.table(stat(count)), 
                                       fill = factor(var36), label = scales::percent(prop.table(stat(count))))
   +geom_bar()
   +scale_fill_brewer(palette = "Set1")
   +geom_text(stat = 'count',
              position = position_dodge(.9), 
              vjust = -0.5, 
              size = 3) + 
     scale_y_continuous(labels = scales::percent)
   +labs(x = ' ', y = 'Proporção',fill="Quantidade"
         ,title="Clientes Insatisfeitos")
   +theme_classic())


grid.arrange(a,b,ncol=2)
X$var36_99=ifelse(X$var36==99,1,0)
X$var36_0=ifelse(X$var36==0,1,0)
teste$var36_99=ifelse(teste$var36==99,1,0)
teste$var36_0=ifelse(teste$var36==0,1,0)

#num_var5
X$num_var5
table(X$num_var5[X$TARGET==0])
table(X$num_var5[X$TARGET==1])
a=(ggplot(X[X$TARGET==0,])+aes(factor(num_var5),y = prop.table(stat(count)), 
                               fill = factor(num_var5), label = scales::percent(prop.table(stat(count))))
   +geom_bar()
   +scale_fill_brewer(palette = "Set1")
   +geom_text(stat = 'count',
              position = position_dodge(.9), 
              vjust = -0.5, 
              size = 3) + 
      scale_y_continuous(labels = scales::percent)
   +labs(x = ' ', y = 'Proporção',fill="Quantidade",title="Clientes Satisfeitos")
   +theme_classic())

b=(ggplot(X[X$TARGET==1,])+aes(factor(num_var5),y = prop.table(stat(count)), 
                               fill = factor(num_var5), label = scales::percent(prop.table(stat(count))))
   +geom_bar()
   +scale_fill_brewer(palette = "Set1")
   +geom_text(stat = 'count',
              position = position_dodge(.9), 
              vjust = -0.5, 
              size = 3) + 
      scale_y_continuous(labels = scales::percent)
   +labs(x = ' ', y = 'Proporção',fill="Quantidade"
         ,title="Clientes Insatisfeitos")
   +theme_classic())
grid.arrange(a,b,ncol=2)

X$num_var5_6=ifelse(X$num_var5==6,1,0)
X$num_var5_0=ifelse(X$num_var5==0,1,0)
teste$num_var5_6=ifelse(teste$num_var5==6,1,0)
teste$num_var5_0=ifelse(teste$num_var5==0,1,0)
#Variável saldo_medio_var5_hace3
summary(X$saldo_medio_var5_hace3)
sort(table(X$saldo_medio_var5_hace3),decreasing=T)[1:5]
(ggplot(X)+aes(saldo_medio_var5_hace3)
   +geom_histogram(aes(y=..density..),color="black",fill="#00AFBB")
   +theme_classic())
(ggplot(X[X$saldo_medio_var5_hace3!=0,])+aes(log(saldo_medio_var5_hace3))
   +geom_histogram(aes(y=..density..),color="black",fill="#00AFBB")
   +theme_classic())

(ggplot(X[X$saldo_medio_var5_hace3!=0,])+aes(log(saldo_medio_var5_hace3))
+geom_histogram(aes(y=..density..),color="black",fill="#00AFBB")
+theme_classic()+facet_grid(~factor(TARGET)))
#Variável saldo_medio_var5_hace2
summary(X$saldo_medio_var5_hace2)
sort(table(X$saldo_medio_var5_hace2),decreasing=T)[1:5]
(ggplot(X)+aes(saldo_medio_var5_hace2)
   +geom_histogram(aes(y=..density..),color="black",fill="#00AFBB")
   +theme_classic())
(ggplot(X[X$saldo_medio_var5_hace2!=0 & X$saldo_medio_var5_hace2!=3,])
   +aes(log(saldo_medio_var5_hace2))
   +geom_histogram(aes(y=..density..),color="black",fill="#00AFBB")
   +theme_classic())

(ggplot(X[X$saldo_medio_var5_hace2!=0,])+aes(log(saldo_medio_var5_hace2))
   +geom_histogram(aes(y=..density..),color="black",fill="#00AFBB")
   +theme_classic()+facet_grid(~factor(TARGET)))

# Variável saldo_medio_var5_ult3 
summary(X$saldo_medio_var5_ult3)
sort(table(X$saldo_medio_var5_ult3),decreasing=T)[1:5]
(ggplot(X[X$saldo_medio_var5_ult3!=0,])+aes(log(saldo_medio_var5_ult3))
   +geom_histogram(aes(y=..density..),color="black",fill="#00AFBB")
   +theme_classic()+facet_grid(~factor(TARGET)))
# Variável saldo_var5
summary(X)
sort(table(X$saldo_var5),decreasing=T)[1:5]

(ggplot(X)+aes(saldo_var5)
   +geom_histogram(aes(y=..density..),color="black",fill="#00AFBB")
   +theme_classic()+facet_grid(~factor(TARGET)))
(ggplot(X)+aes(log(saldo_var5))
   +geom_histogram(aes(y=..density..),color="black",fill="#00AFBB")
   +theme_classic()+facet_grid(~factor(TARGET)))
3
 
#Exportação para a seleção inicial de variáveis
write.csv(X,file="\\Users\\Alberto\\Desktop\\ime-usp\\aprendizagem_estatistica\\trabalho_aplicacao\\treino_com_criacao_variaveis.csv",row.names=F)
write.csv(teste,file="\\Users\\Alberto\\Desktop\\ime-usp\\aprendizagem_estatistica\\trabalho_aplicacao\\teste_com_criacao_variaveis.csv",row.names=F)




#Visualização das métricas após a medição das métricas
metricas=read.csv(file.choose())
head(metricas)

a=(ggplot(metricas)
   +geom_line(aes(x=indices,y=roc_auc_treino,col="Treino"),size=1.7)
   +geom_line(aes(x=indices,y=roc_auc_teste,col="Teste"),size=1.7)
   +labs(color='Conjuntos', y="ROC AUC ",x="Quantidade de variáveis"
         ,title="Área sobre a curva ROC")
   +scale_colour_brewer(palette = "Set1")
   +theme_classic())

b=(ggplot(metricas)
   +geom_line(aes(x=indices,y=recall_0_treino,col="Treino"),size=1.7)
   +geom_line(aes(x=indices,y=recall_0_teste,col="Teste"),size=1.7)
   +labs(color='Conjuntos', y="Recall",x="Quantidade de variáveis"
         ,title="Recall-Clientes Insatisfeitos")
   +scale_colour_brewer(palette = "Set1")
   +theme_classic())


c=(ggplot(metricas)
   +geom_line(aes(x=indices,y=recall_1_treino,col="Treino"),size=1.7)
   +geom_line(aes(x=indices,y=recall_1_teste,col="Teste"),size=1.7)
   +labs(color='Conjuntos', y="Recall",x="Quantidade de variáveis"
         ,title="Recall-Clientes Insatisfeitos")
   +scale_colour_brewer(palette = "Set1")
   +theme_classic())

d=(ggplot(metricas)
   +geom_line(aes(x=indices,y=precision_0_treino,col="Treino"),size=1.7)
   +geom_line(aes(x=indices,y=precision_0_teste,col="Teste"),size=1.7)
   +labs(color='Conjuntos', y="Precision",x="Quantidade de variáveis"
         ,title="Precision-Clientes Satisfeitos")
   +scale_colour_brewer(palette = "Set1")
   +theme_classic())

e=(ggplot(metricas)
   +geom_line(aes(x=indices,y=precision_1_treino,col="Treino"),size=1.7)
   +geom_line(aes(x=indices,y=precision_1_teste,col="Teste"),size=1.7)
   +labs(color='Conjuntos', y="Precision",x="Quantidade de variáveis"
         ,title="Precision-Clientes Insatisfeitos")
   +scale_colour_brewer(palette = "Set1")
   +theme_classic())

f=(ggplot(metricas)
   +geom_line(aes(x=indices,y=f1_0_treino,col="Treino"),size=1.7)
   +geom_line(aes(x=indices,y=f1_0_teste,col="Teste"),size=1.7)
   +labs(color='Conjuntos', y="F1 Score",x="Quantidade de variáveis"
         , title="F1 Score-Clientes Satisfeitos")
   +scale_colour_brewer(palette = "Set1")
   +theme_classic())

g=(ggplot(metricas)
   +geom_line(aes(x=indices,y=f1_1_treino,col="Treino"),size=1.7)
   +geom_line(aes(x=indices,y=f1_1_teste,col="Teste"),size=1.7)
   +labs(color='Conjuntos', y="F1 Score",x="Quantidade de variáveis", 
         title="F1 Score-Clientes Insatisfeitos")
   +scale_colour_brewer(palette = "Set1")
   +theme_classic())

h=(ggplot(metricas)
   +geom_line(aes(x=indices,y=f1_treino,col="Treino"),size=1.7)
   +geom_line(aes(x=indices,y=f1_teste,col="Teste"),size=1.7)
   +labs(y="F1 Score",x="Quantidade de variáveis")
   +scale_colour_brewer(palette = "Set1")
   +labs(color='Conjuntos', title="F1 Score médio das duas classes")
   +theme_classic())
grid.arrange(c,e, ncol=2)
grid.arrange(g,h, ncol=2)

head(metricas)
metricas[,c("indices","recall_1_teste","precision_1_teste","f1_1_teste","f1_teste")]
76000-17468-2406
#Metricas dos melhores modelos
metricas_boosting=read.csv(file.choose())
head(metricas_boosting)
indices=1:3
a2=(ggplot(metricas_boosting)
   +geom_line(aes(x=indices,y=roc_auc_modelo1,col="AdaBoost"),size=1.7)
   +geom_line(aes(x=indices,y=roc_auc_modelo2,col="GradientBoosting"),size=1.7)
   +labs(color='Modelos', y="ROC AUC ",x="Partição"
         ,title="Área sobre a curva ROC")
   +scale_colour_brewer(palette = "Set1")
   +theme_classic())

b2=(ggplot(metricas_boosting)
   +geom_line(aes(x=indices,y=recall_0_modelo1,col="AdaBoost"),size=1.7)
   +geom_line(aes(x=indices,y=recall_0_modelo2,col="GradientBoosting"),size=1.7)
   +labs(color='Modelos', y="Recall",x="Partição"
         ,title="Recall-Clientes Satisfeitos")
   +scale_colour_brewer(palette = "Set1")
   +theme_classic())


c2=(ggplot(metricas_boosting)
   +geom_line(aes(x=indices,y=recall_1_modelo1,col="AdaBoost"),size=1.7)
   +geom_line(aes(x=indices,y=recall_1_modelo2,col="GradientBoosting"),size=1.7)
   +labs(color='Modelos', y="Recall",x="Partição"
         ,title="Recall-Clientes Insatisfeitos")
   +scale_colour_brewer(palette = "Set1")
   +theme_classic())

d2=(ggplot(metricas_boosting)
   +geom_line(aes(x=indices,y=precision_0_modelo1,col="AdaBoost"),size=1.7)
   +geom_line(aes(x=indices,y=precision_0_modelo2,col="GradientBoosting"),size=1.7)
   +labs(color='Modelos', y="Precision",x="Partição"
         ,title="Precision-Clientes Satisfeitos")
   +scale_colour_brewer(palette = "Set1")
   +theme_classic())

e2=(ggplot(metricas_boosting)
   +geom_line(aes(x=indices,y=precision_1_modelo1,col="AdaBoost"),size=1.7)
   +geom_line(aes(x=indices,y=precision_1_modelo2,col="GradientBoosting"),size=1.7)
   +labs(color='Modelos', y="Precision",x="Partição"
         ,title="Precision-Clientes Insatisfeitos")
   +scale_colour_brewer(palette = "Set1")
   +theme_classic())

f2=(ggplot(metricas_boosting)
   +geom_line(aes(x=indices,y=f1_0_modelo1,col="AdaBoost"),size=1.7)
   +geom_line(aes(x=indices,y=f1_0_modelo2,col="GradientBoosting"),size=1.7)
   +labs(color='Modelos', y="F1 Score",x="Partição"
         , title="F1 Score-Clientes Satisfeitos")
   +scale_colour_brewer(palette = "Set1")
   +theme_classic())

g2=(ggplot(metricas_boosting)
   +geom_line(aes(x=indices,y=f1_1_modelo1,col="AdaBoost"),size=1.7)
   +geom_line(aes(x=indices,y=f1_1_modelo2,col="GradientBoosting"),size=1.7)
   +labs(color='Modelos', y="F1 Score",x="Partição", 
         title="F1 Score-Clientes Insatisfeitos")
   +scale_colour_brewer(palette = "Set1")
   +theme_classic())

h2=(ggplot(metricas_boosting)
   +geom_line(aes(x=indices,y=f1_modelo1,col="AdaBoost"),size=1.7)
   +geom_line(aes(x=indices,y=f1_modelo2,col="GradientBoosting"),size=1.7)
   +labs(y="F1 Score",x="Partição")
   +scale_colour_brewer(palette = "Set1")
   +labs(color='Modelos', title="F1 Score médio das duas classes")
   +theme_classic())
grid.arrange(c2,e2,g2,h2, ncol=2)
#grid.arrange(, ncol=2)

metricas_modelo1=colMeans(metricas_boosting[,c("recall_0_modelo1",
                                               "recall_1_modelo1"
,"precision_0_modelo1", "precision_1_modelo1", "f1_0_modelo1"
,"f1_1_modelo1", "f1_modelo1")])

metricas_modelo2=colMeans(metricas_boosting[,c("recall_0_modelo2",
                                               "recall_1_modelo2"
      ,"precision_0_modelo2", "precision_1_modelo2", "f1_0_modelo2"
               ,"f1_1_modelo2", "f1_modelo2")])

require(xtable)
xtable(cbind.data.frame(metricas_modelo1, metricas_modelo2),digits=4)
head(metricas_boosting)
head(metricas)
