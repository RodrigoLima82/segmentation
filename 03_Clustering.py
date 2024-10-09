# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/segmentation.git. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/customer-segmentation.

# COMMAND ----------

# MAGIC %md O objetivo deste notebook é identificar segmentos potenciais para nossos domicílios usando uma técnica de agrupamento.

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

import numpy as np
import pandas as pd

import mlflow
import os

from delta.tables import *

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import seaborn as sns

# COMMAND ----------

# MAGIC %md ## Passo 1: Recuperar Características
# MAGIC
# MAGIC Seguindo o trabalho realizado em nosso último notebook, nossos domicílios agora são identificados por um número limitado de características que capturam a variação encontrada em nosso conjunto de características original. Podemos recuperar essas características da seguinte forma:

# COMMAND ----------

# DBTITLE 1,Initialize the gold table paths
dbutils.fs.rm("/tmp/completejourney/gold/", True)
dbutils.fs.mkdirs("/tmp/completejourney/gold/")

# COMMAND ----------

# DBTITLE 1,Retrieve Transformed Features
# retrieve household (transformed) features
household_X_pd = spark.table('rodrigo_catalog.journey.features_finalized').toPandas()

# remove household ids from dataframe
X = household_X_pd.drop(['household_id'], axis=1)

household_X_pd

# COMMAND ----------

# MAGIC %md O significado exato de cada recurso é muito difícil de articular devido às complexas transformações utilizadas em sua engenharia. Ainda assim, eles podem ser usados para realizar clustering. (Através de um perfil que iremos realizar em nosso próximo notebook, poderemos obter insights sobre a natureza de cada cluster.)
# MAGIC
# MAGIC Como primeiro passo, vamos visualizar nossos dados para ver se algum agrupamento natural se destaca. Como estamos trabalhando com um espaço hiperdimensional, não podemos visualizar perfeitamente nossos dados, mas com uma representação em 2D (usando nossos dois primeiros recursos principais), podemos ver que há um grande cluster considerável em nossos dados e potencialmente alguns clusters adicionais, mais vagamente organizados:

# COMMAND ----------

# DBTITLE 1,Plot Households
fig, ax = plt.subplots(figsize=(10,8))

_ = sns.scatterplot(
  data=X,
  x='Dim.1',
  y='Dim.2',
  alpha=0.5,
  ax=ax
  )

# COMMAND ----------

# MAGIC %md ## Passo 2: Agrupamento K-Means
# MAGIC
# MAGIC Nossa primeira tentativa de agrupamento fará uso do algoritmo K-means. O K-means é um algoritmo simples e popular para dividir instâncias em clusters ao redor de um número pré-definido de *centróides* (centros de cluster). O algoritmo funciona gerando um conjunto inicial de pontos dentro do espaço para servir como centros de cluster. As instâncias são então associadas ao ponto mais próximo desses pontos para formar um cluster, e o verdadeiro centro do cluster resultante é recalculado. Os novos centróides são então usados para reagrupar os membros do cluster, e o processo é repetido até que uma solução estável seja gerada (ou até que o número máximo de iterações seja esgotado). Uma rápida execução de demonstração do algoritmo pode produzir um resultado como o seguinte:

# COMMAND ----------

# DBTITLE 1,Demonstrate Cluster Assignment
# set up the experiment that mlflow logs runs to: an experiment in the user's personal workspace folder
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = f"/Users/{useremail}/segmentation_rodrigo"
mlflow.set_experiment(experiment_name) 

# initial cluster count
initial_n = 4

# train the model
initial_model = KMeans(
  n_clusters=initial_n,
  max_iter=1000
  )

# fit and predict per-household cluster assignment
init_clusters = initial_model.fit_predict(X)

# combine households with cluster assignments
labeled_X_pd = (
  pd.concat( 
    [X, pd.DataFrame(init_clusters,columns=['cluster'])],
    axis=1
    )
  )

# visualize cluster assignments
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
  data=labeled_X_pd,
  x='Dim.1',
  y='Dim.2',
  hue='cluster',
  palette=[cm.nipy_spectral(float(i) / initial_n) for i in range(initial_n)],
  legend='brief',
  alpha=0.5,
  ax = ax
  )
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------

# MAGIC %md Nossa execução inicial do modelo demonstra a mecânica de gerar uma solução de agrupamento K-means, mas também demonstra algumas das limitações desse método. Primeiro, precisamos especificar o número de clusters. Definir o valor incorretamente pode resultar na criação de vários clusters menores ou apenas alguns clusters maiores, o que pode não refletir a estrutura mais imediata e natural dos dados.
# MAGIC
# MAGIC Segundo, os resultados do algoritmo dependem muito dos centróides com os quais ele é inicializado. O uso do algoritmo de inicialização K-means++ aborda alguns desses problemas, garantindo melhor que os centróides iniciais estejam dispersos por todo o espaço populado, mas ainda há um elemento de aleatoriedade nessas seleções que pode ter grandes consequências para nossos resultados.
# MAGIC
# MAGIC Para começar a lidar com esses desafios, geraremos um grande número de execuções do modelo em uma variedade de contagens de clusters potenciais. Para cada execução, calcularemos a soma dos quadrados das distâncias entre os membros e os centróides atribuídos (*inertia*), bem como uma métrica secundária (*silhouette score*) que fornece uma medida combinada de coesão inter-cluster e separação intra-cluster (variando entre -1 e 1, sendo valores mais altos melhores). Devido ao grande número de iterações que faremos, distribuiremos esse trabalho em nosso cluster Databricks para que possa ser concluído em tempo hábil:
# MAGIC
# MAGIC **NOTA** Estamos usando um RDD do Spark como uma técnica simples para pesquisar exaustivamente nosso espaço de parâmetros de maneira distribuída. Essa é uma técnica comumente usada para pesquisas eficientes em uma faixa definida de valores.

# COMMAND ----------

# DBTITLE 1,Iterate over Potential Values of K
# broadcast features so that workers can access efficiently
X_broadcast = sc.broadcast(X)

# function to train model and return metrics
def evaluate_model(n):
  model = KMeans( n_clusters=n, init='k-means++', n_init=1, max_iter=10000)
  clusters = model.fit(X_broadcast.value).labels_
  return n, float(model.inertia_), float(silhouette_score(X_broadcast.value, clusters))


# define number of iterations for each value of k being considered
iterations = (
  spark
    .range(100) # iterations per value of k
    .crossJoin( spark.range(2,21).withColumnRenamed('id','n')) # cluster counts
    .repartition(sc.defaultParallelism)
    .select('n')
    .rdd
    )

# train and evaluate model for each iteration
results_pd = (
  spark
    .createDataFrame(
      iterations.map(lambda n: evaluate_model(n[0])), # iterate over each value of n
      schema=['n', 'inertia', 'silhouette']
      ).toPandas()
    )

# remove broadcast set from workers
X_broadcast.unpersist()

# COMMAND ----------

# MAGIC %md Plotando a inércia em relação a n, ou seja, o número alvo de clusters, podemos ver que a soma total das distâncias quadradas entre os membros do cluster e os centros do cluster diminui à medida que aumentamos o número de clusters em nossa solução. Nosso objetivo não é reduzir a inércia a zero (o que seria alcançado se tornássemos cada membro o centro de seu próprio cluster com apenas 1 membro), mas sim identificar o ponto na curva onde a queda incremental na inércia é diminuída. Em nosso gráfico, podemos identificar esse ponto como ocorrendo em algum lugar entre 2 e 6 clusters:

# COMMAND ----------

# DBTITLE 1,Inertia over Cluster Count
display(results_pd)

# COMMAND ----------

# MAGIC %md Interpretar o gráfico do cotovelo ou gráfico de inércia em relação a n é bastante subjetivo, e, como tal, pode ser útil examinar como outra métrica se comporta em relação ao número de clusters. Plotar o escore de silhueta em relação a n nos oferece a oportunidade de identificar um pico (joelho) além do qual o escore diminui. O desafio, como antes, é determinar exatamente a localização desse pico, especialmente considerando que os escores de silhueta para nossas iterações variam muito mais do que nossos escores de inércia:

# COMMAND ----------

# MAGIC %md Embora forneça uma segunda perspectiva, o gráfico de escores de silhueta reforça a ideia de que a seleção do número de clusters para o K-means é um pouco subjetiva. O conhecimento de domínio aliado às informações desses e de outros gráficos semelhantes (como um gráfico da [estatística de lacuna](https://towardsdatascience.com/k-means-clustering-and-the-gap-statistics-4c5d414acd29)) podem ajudar a apontar para um número ótimo de clusters, mas até o momento não existem meios objetivos amplamente aceitos para determinar esse valor.
# MAGIC
# MAGIC **NOTA** Precisamos ter cuidado para evitar buscar o valor mais alto para o escore de silhueta no gráfico do joelho. Escores mais altos podem ser obtidos com valores mais altos de n, simplesmente empurrando os outliers para clusters trivialmente pequenos.
# MAGIC
# MAGIC Para o nosso modelo, vamos escolher um valor de 2. Ao olhar o gráfico de inércia, parece haver evidências que suportam esse valor. Ao examinar os escores de silhueta, a solução de clusterização parece ser muito mais estável nesse valor do que em valores mais baixos da faixa. Para obter conhecimento de domínio, podemos conversar com nossos especialistas em promoções e obter sua perspectiva não apenas sobre como diferentes domicílios respondem às promoções, mas também sobre qual número de clusters pode ser viável a partir desse exercício. Mas, o mais importante, a presença de 2 clusters bem separados parece naturalmente se destacar em nossa visualização.
# MAGIC
# MAGIC Com um valor para n identificado, agora precisamos gerar um design final de clusters. Dada a aleatoriedade dos resultados que obtemos de uma execução do K-means (conforme capturado nos escores de silhueta amplamente variáveis), podemos adotar uma abordagem de *melhor-de-k* para definir nosso modelo de cluster. Nessa abordagem, executamos várias vezes o modelo K-means e selecionamos a execução que oferece o melhor resultado, medido por uma métrica como o escore de silhueta. Para distribuir esse trabalho, implementaremos uma função personalizada que nos permitirá atribuir a cada trabalhador a tarefa de encontrar uma solução melhor-de-k e, em seguida, selecionar a melhor solução entre os resultados desse trabalho:
# MAGIC
# MAGIC **NOTA** Estamos usando novamente um RDD para nos permitir distribuir o trabalho em nosso cluster. O RDD *iterations* conterá um valor para cada iteração a ser realizada. Usando *mapPartitions()*, determinaremos quantas iterações são atribuídas a uma determinada partição e, em seguida, forçaremos esse trabalhador a realizar uma avaliação melhor-de-k adequadamente configurada. Cada partição enviará de volta o melhor modelo que pôde descobrir e, em seguida, selecionaremos o melhor entre eles.

# COMMAND ----------

# DBTITLE 1,Identify Best of K Model
total_iterations = 50000
n_for_bestofk = 2 
X_broadcast = sc.broadcast(X)

def find_bestofk_for_partition(partition):
   
  # count iterations in this partition
  n_init = sum(1 for i in partition)
  
  # perform iterations to get best of k
  model = KMeans( n_clusters=n_for_bestofk, n_init=n_init, init='k-means++', max_iter=10000)
  model.fit(X_broadcast.value)
  
  # score model
  score = float(silhouette_score(X_broadcast.value, model.labels_))
  
  # return (score, model)
  yield (score, model)


# build RDD for distributed iteration
iterations = sc.range(
              total_iterations, 
              numSlices= sc.defaultParallelism * 4
              ) # distribute work into fairly even number of partitions that allow us to track progress
                        
# retrieve best of distributed iterations
bestofk_results = (
  iterations
    .mapPartitions(find_bestofk_for_partition)
    .sortByKey(ascending=False)
    .take(1)
    )[0]

# get score and model
bestofk_score = bestofk_results[0]
bestofk_model = bestofk_results[1]
bestofk_clusters = bestofk_model.labels_

# print best score obtained
print('Silhouette Score: {0:.6f}'.format(bestofk_score))

# combine households with cluster assignments
bestofk_labeled_X_pd = (
  pd.concat( 
    [X, pd.DataFrame(bestofk_clusters,columns=['cluster'])],
    axis=1
    )
  )
                        
# clean up 
X_broadcast.unpersist()

# COMMAND ----------

# MAGIC %md Agora podemos visualizar nossos resultados para ter uma ideia de como os clusters se alinham com a estrutura dos nossos dados:

# COMMAND ----------

# DBTITLE 1,Visualize Best of K Clusters
# visualize cluster assignments
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
  data=bestofk_labeled_X_pd,
  x='Dim.1',
  y='Dim.2',
  hue='cluster',
  palette=[cm.nipy_spectral(float(i) / n_for_bestofk) for i in range(n_for_bestofk)],  # align colors with those used in silhouette plots
  legend='brief',
  alpha=0.5,
  ax = ax
  )
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------

# MAGIC %md Os resultados da nossa análise não são surpreendentes, mas não precisam ser. Nossos dados indicam que, para essas características, poderíamos considerar razoavelmente nossos clientes em dois grupos bastante distintos. No entanto, pode ser interessante analisar como cada cliente se encaixa nesses grupos, o que podemos fazer por meio de um gráfico de silhueta por instância:
# MAGIC
# MAGIC **NOTA** Este código representa uma versão modificada dos [gráficos de silhueta](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html) fornecidos na documentação do Sci-Kit Learn.

# COMMAND ----------

# DBTITLE 1,Examine Per-Member Silhouette Scores
# modified from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

def plot_silhouette_chart(features, labels):
  
  n = len(np.unique(labels))
  
  # configure plot area
  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(8, 5)

  # configure plots for silhouette scores between -1 and 1
  ax.set_xlim([-0.1, 1])
  ax.set_ylim([0, len(features) + (n + 1) * 10])
  
  # avg silhouette score
  score = silhouette_score(features, labels)

  # compute the silhouette scores for each sample
  sample_silhouette_values = silhouette_samples(features, labels)

  y_lower = 10

  for i in range(n):

      # get and sort members by cluster and score
      ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
      ith_cluster_silhouette_values.sort()

      # size y based on sample count
      size_cluster_i = ith_cluster_silhouette_values.shape[0]
      y_upper = y_lower + size_cluster_i

      # pretty up the charts
      color = cm.nipy_spectral(float(i) / n)
      
      ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

      # label the silhouette plots with their cluster numbers at the middle
      ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

      # compute the new y_lower for next plot
      y_lower = y_upper + 10  # 10 for the 0 samples


  ax.set_title("Average silhouette of {0:.3f} with {1} clusters".format(score, n))
  ax.set_xlabel("The silhouette coefficient values")
  ax.set_ylabel("Cluster label")

  # vertical line for average silhouette score of all the values
  ax.axvline(x=score, color="red", linestyle="--")

  ax.set_yticks([])  # clear the yaxis labels / ticks
  ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
  
  return fig, ax

_ = plot_silhouette_chart(X, bestofk_clusters)

# COMMAND ----------

# MAGIC %md A partir do gráfico de silhueta, parece que temos um cluster um pouco maior que o outro. Esse cluster parece ser razoavelmente coerente. Nossos outros clusters parecem estar um pouco mais dispersos, com uma queda mais rápida nos valores de pontuação de silhueta, levando alguns membros a terem pontuações de silhueta negativas (indicando sobreposição com outro cluster).
# MAGIC
# MAGIC Essa solução pode ser útil para entender melhor o comportamento do cliente em relação a ofertas promocionais. Vamos persistir nossas atribuições de cluster antes de examinar outras técnicas de clustering:

# COMMAND ----------

# DBTITLE 1,Persist Cluster Assignments
# persist household id and cluster assignment
( 
  spark # bring together household and cluster ids
    .createDataFrame(
       pd.concat( 
          [household_X_pd, pd.DataFrame(bestofk_clusters,columns=['bestofk_cluster'])],
          axis=1
          )[['household_id','bestofk_cluster']]   
      )
    .write  # write data to delta 
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable('rodrigo_catalog.journey.household_clusters')
  )

# COMMAND ----------

# MAGIC %md ## Passo 3: Agrupamento Hierárquico
# MAGIC
# MAGIC Além do K-means, as técnicas de agrupamento hierárquico são frequentemente usadas em exercícios de segmentação de clientes. Com as variantes aglomerativas dessas técnicas, os clusters são formados ligando os membros mais próximos uns dos outros e, em seguida, ligando esses clusters para formar clusters de nível superior até que um único cluster que englobe todos os membros do conjunto seja formado.
# MAGIC
# MAGIC Ao contrário do K-means, o processo aglomerativo é determinístico, de modo que execuções repetidas no mesmo conjunto de dados levam ao mesmo resultado de agrupamento. Portanto, embora as técnicas de agrupamento hierárquico sejam frequentemente criticadas por serem mais lentas que o K-means, o tempo total de processamento para chegar a um resultado específico pode ser reduzido, pois não são necessárias execuções repetidas do algoritmo para chegar a um resultado "melhor".
# MAGIC
# MAGIC Para ter uma melhor compreensão de como essa técnica funciona, vamos treinar uma solução de agrupamento hierárquico e visualizar sua saída:

# COMMAND ----------

# DBTITLE 1,Function to Plot Dendrogram
# modified from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py

# function to generate dendrogram
def plot_dendrogram(model, **kwargs):

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
                      [model.children_, 
                       model.distances_,
                       counts]
                      ).astype(float)

    # Plot the corresponding dendrogram
    j = 5
    set_link_color_palette(
      [matplotlib.colors.rgb2hex(cm.nipy_spectral(float(i) / j)) for i in range(j)]
      )
    dendrogram(linkage_matrix, **kwargs)

# COMMAND ----------

# DBTITLE 1,Train & Visualize Hierarchical Model
# train cluster model
inithc_model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward')
inithc_model.fit(X)

# generate visualization
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(15, 8)

plot_dendrogram(inithc_model, truncate_mode='level', p=6) # 6 levels max
plt.title('Hierarchical Clustering Dendrogram')
_ = plt.xlabel('Number of points in node (or index of point if no parenthesis)')

# COMMAND ----------

# MAGIC %md O dendrograma é lido de baixo para cima. Cada ponto inicial representa um cluster composto por um certo número de membros. Todo o processo pelo qual esses membros se juntam para formar esses clusters específicos não é visualizado (embora você possa ajustar o argumento *p* na função *plot_dendrogram* para ver mais detalhes do processo).
# MAGIC
# MAGIC Conforme você sobe no dendrograma, os clusters convergem para formar novos clusters. O comprimento vertical percorrido para atingir esse ponto de convergência nos diz algo sobre a distância entre esses clusters. Quanto maior o comprimento, maior é a lacuna entre os clusters convergentes.
# MAGIC
# MAGIC O dendrograma nos dá uma ideia de como a estrutura geral do conjunto de dados se forma, mas não nos direciona para um número específico de clusters para nossa solução final de clustering. Para isso, precisamos recorrer à plotagem de uma métrica, como os escores de silhueta, para identificar o número apropriado de clusters para nossa solução.
# MAGIC
# MAGIC Antes de plotar a silhueta em relação a vários números de clusters, é importante examinar os meios pelos quais os clusters são combinados para formar novos clusters. Existem muitos algoritmos (*linkages*) para isso. A biblioteca SciKit-Learn suporta quatro deles. São eles:
# MAGIC <p>
# MAGIC * *ward* - combina clusters de forma que a soma das distâncias quadradas dentro dos clusters recém-formados seja minimizada
# MAGIC * *average* - combina clusters com base na distância média entre todos os pontos nos clusters
# MAGIC * *single* - combina clusters com base na distância mínima entre quaisquer dois pontos nos clusters
# MAGIC * *complete* - combina clusters com base na distância máxima entre quaisquer dois pontos nos clusters
# MAGIC
# MAGIC Mecanismos de ligação diferentes podem resultar em resultados de clustering muito diferentes. O método de Ward (denotado pelo mecanismo de ligação *ward*) é considerado o mais adequado para a maioria dos exercícios de clustering, a menos que o conhecimento do domínio indique o uso de um método alternativo:

# COMMAND ----------

# DBTITLE 1,Identify Number of Clusters
results = []

# train models with n number of clusters * linkages
for a in ['ward']:  # linkages
  for n in range(2,21): # evaluate 2 to 20 clusters

    # fit the algorithm with n clusters
    model = AgglomerativeClustering(n_clusters=n, linkage=a)
    clusters = model.fit(X).labels_

    # capture the inertia & silhouette scores for this value of n
    results += [ (n, a, silhouette_score(X, clusters)) ]

results_pd = pd.DataFrame(results, columns=['n', 'linkage', 'silhouette'])
display(results_pd)

# COMMAND ----------

# MAGIC %md Os resultados indicam que nossos melhores resultados podem ser encontrados usando 5 clusters:

# COMMAND ----------

# DBTITLE 1,Train & Evaluate Model
n_for_besthc = 5
linkage_for_besthc = 'ward'
 
# configure model
besthc_model = AgglomerativeClustering( n_clusters=n_for_besthc, linkage=linkage_for_besthc)

# train and predict clusters
besthc_clusters = besthc_model.fit(X).labels_

# score results
besthc_score = silhouette_score(X, besthc_clusters)

# print best score obtained
print('Silhouette Score: {0:.6f}'.format(besthc_score))

# combine households with cluster assignments
besthc_labeled_X_pd = (
  pd.concat( 
    [X, pd.DataFrame(besthc_clusters,columns=['cluster'])],
    axis=1
    )
  )

# COMMAND ----------

# MAGIC %md Ao visualizar esses clusters, podemos ver como os agrupamentos estão distribuídos dentro da estrutura dos dados. Em nossa visualização inicial das características, argumentamos que havia dois clusters de alto nível que se destacavam (e nosso algoritmo de K-means parece ter captado isso muito bem). Aqui, nosso algoritmo de agrupamento hierárquico parece ter captado melhor os subclusters mais soltos, embora também tenha captado algumas residências pouco organizadas para um cluster muito pequeno:

# COMMAND ----------

# DBTITLE 1,Visualize Clusters
# visualize cluster assignments
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
  data=besthc_labeled_X_pd,
  x='Dim.1',
  y='Dim.2',
  hue='cluster',
  palette=[cm.nipy_spectral(float(i) / n_for_besthc) for i in range(n_for_besthc)],  # align colors with those used in silhouette plots
  legend='brief',
  alpha=0.5,
  ax = ax
  )
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------

# MAGIC %md Nossos escores de silhueta por instância mostram que temos um pouco mais de sobreposição entre os clusters quando examinados nesse nível. Um dos clusters tem tão poucos membros que não parece valer a pena mantê-lo, especialmente quando revisamos a visualização em 2D e vemos que esses pontos parecem estar altamente misturados com outros clusters (pelo menos quando vistos dessa perspectiva):

# COMMAND ----------

# DBTITLE 1,Examine Per-Member Silhouette Scores
_ = plot_silhouette_chart(X, besthc_clusters)

# COMMAND ----------

# MAGIC %md Com isso em mente, vamos reajustar nosso modelo com uma contagem de cluster de 4 e, em seguida, persistir esses resultados:

# COMMAND ----------

# DBTITLE 1,ReTrain & Evaluate Model
n_for_besthc = 4
linkage_for_besthc = 'ward'
 
# configure model
besthc_model = AgglomerativeClustering( n_clusters=n_for_besthc, linkage=linkage_for_besthc)

# train and predict clusters
besthc_clusters = besthc_model.fit(X).labels_

# score results
besthc_score = silhouette_score(X, besthc_clusters)

# print best score obtained
print('Silhouette Score: {0:.6f}'.format(besthc_score))

# combine households with cluster assignments
besthc_labeled_X_pd = (
  pd.concat( 
    [X, pd.DataFrame(besthc_clusters,columns=['cluster'])],
    axis=1
    )
  )

# COMMAND ----------

# DBTITLE 1,Visualize Clusters
# visualize cluster assignments
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
  data=besthc_labeled_X_pd,
  x='Dim.1',
  y='Dim.2',
  hue='cluster',
  palette=[cm.nipy_spectral(float(i) / n_for_besthc) for i in range(n_for_besthc)],  # align colors with those used in silhouette plots
  legend='brief',
  alpha=0.5,
  ax = ax
  )
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------

# DBTITLE 1,Examine Per-Member Silhouette Scores
_ = plot_silhouette_chart(X, besthc_clusters)

# COMMAND ----------

# DBTITLE 1,Add Field to Hold Hierarchical Cluster Assignment
# add column to previously created table to allow assignment of cluster ids
# try/except used here in case this statement is being rurun against a table with field already in place
try:
  spark.sql('ALTER TABLE rodrigo_catalog.journey.household_clusters ADD COLUMN (hc_cluster integer)')
except:
  pass  

# COMMAND ----------

# DBTITLE 1,Update Persisted Data to Hold Hierarchical Cluster Assignment
# assemble household IDs and new cluster IDs
updates = (
  spark
    .createDataFrame(
       pd.concat( 
          [household_X_pd, pd.DataFrame(besthc_clusters,columns=['hc_cluster'])],
          axis=1
          )[['household_id','hc_cluster']]   
      )
  )

# merge new cluster ID data with existing table  
deltaTable = DeltaTable.forName(spark, 'rodrigo_catalog.journey.household_clusters')

(
  deltaTable.alias('target')
    .merge(
      updates.alias('source'),
      'target.household_id=source.household_id'
      )
    .whenMatchedUpdate(set = { 'hc_cluster' : 'source.hc_cluster' } )
    .execute()
  )

# COMMAND ----------

# MAGIC %md ## Passo 4: Outras Técnicas
# MAGIC
# MAGIC Apenas começamos a explorar as técnicas de agrupamento disponíveis para nós. [K-Medoids](https://scikit-learn-extra.readthedocs.io/en/latest/generated/sklearn_extra.cluster.KMedoids.html), uma variação do K-means que centraliza os clusters em membros reais do conjunto de dados, permite considerar métodos alternativos (além da distância euclidiana) para avaliar a similaridade entre membros e pode ser mais robusto a ruídos e outliers em um conjunto de dados. [Density-Based Spatial Clustering of Applications with Noise (DBSCAN)](https://scikit-learn.org/stable/modules/clustering.html#dbscan) é outra técnica interessante de agrupamento que identifica clusters em áreas de alta densidade de membros, ignorando membros dispersos em regiões de baixa densidade. Essa técnica parece ser adequada para este conjunto de dados, mas ao examinar o DBSCAN (não mostrado), tivemos dificuldade em ajustar os parâmetros *epsilon* e *contagem mínima de amostras* (que controlam como as regiões de alta densidade são identificadas) para obter uma solução de agrupamento de alta qualidade. E [Gaussian Mixture Models](https://scikit-learn.org/stable/modules/mixture.html#gaussian-mixture-models) oferecem ainda outra abordagem popular em exercícios de segmentação, permitindo a formação de clusters com formas não esféricas de maneira mais fácil.
# MAGIC
# MAGIC Além de algoritmos alternativos, há um trabalho emergente no desenvolvimento de modelos de conjunto de clusters (também conhecidos como *consensus clustering*). Introduzido pela primeira vez por [Monti *et al.*](https://link.springer.com/article/10.1023/A:1023949509487) para aplicação em pesquisas genômicas, o consensus clustering tem se tornado popular em uma ampla gama de aplicações nas ciências da vida, embora pareça haver pouca adoção até o momento na área de segmentação de clientes. O suporte para consensus clustering por meio dos pacotes [OpenEnsembles](https://www.jmlr.org/papers/v19/18-100.html) e [kemlglearn](https://nbviewer.jupyter.org/github/bejar/URLNotebooks/blob/master/Notebooks/12ConsensusClustering.ipynb) está disponível em Python, embora um suporte mais robusto para consensus clustering possa ser encontrado em bibliotecas R, como [diceR](https://cran.r-project.org/web/packages/diceR/index.html). Uma exploração limitada desses pacotes e bibliotecas (não mostrada) produziu resultados mistos, embora suspeitemos que isso tenha mais a ver com nossos próprios desafios de ajuste de hiperparâmetros e menos com os algoritmos em si.
