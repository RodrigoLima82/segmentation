# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/segmentation.git. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/customer-segmentation.

# COMMAND ----------

# MAGIC %md O objetivo deste notebook é entender melhor os clusters gerados no notebook anterior, utilizando algumas técnicas de perfilagem padrão.

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import mlflow

import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.graphics.mosaicplot import mosaic

import math

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from pyspark.sql.functions import expr

# COMMAND ----------

# MAGIC %md ## Passo 1: Montar o Conjunto de Dados Segmentado
# MAGIC
# MAGIC Agora temos clusters, mas não temos uma compreensão clara do que exatamente eles representam. O trabalho de engenharia de recursos que realizamos para evitar problemas com os dados que poderiam levar a soluções inválidas ou inadequadas tornou os dados muito difíceis de interpretar.
# MAGIC
# MAGIC Para resolver esse problema, iremos recuperar os rótulos dos clusters (atribuídos a cada domicílio) juntamente com os recursos originais associados a cada um:

# COMMAND ----------

# DBTITLE 1,Retrieve Features & Labels
# retrieve features and labels
spark.sql("USE journey")
household_basefeatures = spark.table('rodrigo_catalog.journey.household_features')
household_finalfeatures = spark.table('rodrigo_catalog.journey.features_finalized')
labels = spark.table('rodrigo_catalog.journey.household_clusters')

# assemble labeled feature sets
labeled_basefeatures_pd = (
  labels
    .join(household_basefeatures, on='household_id')
  ).toPandas()

labeled_finalfeatures_pd = (
  labels
    .join(household_finalfeatures, on='household_id')
  ).toPandas()

# get name of all non-feature columns
label_columns = labels.columns

labeled_basefeatures_pd

# COMMAND ----------

# MAGIC %md Antes de prosseguir com nossa análise desses dados, vamos definir algumas variáveis que serão usadas para controlar o restante de nossa análise. Temos vários designs de cluster, mas para este notebook, vamos focar nossa atenção nos resultados do nosso modelo de agrupamento hierárquico:

# COMMAND ----------

# DBTITLE 1,Set Cluster Design to Analyze
cluster_column = 'hc_cluster'
cluster_count = len(np.unique(labeled_finalfeatures_pd[cluster_column]))
cluster_colors = [cm.nipy_spectral(float(i)/cluster_count) for i in range(cluster_count)]

# COMMAND ----------

# MAGIC %md ## Passo 2: Perfilar Segmentos
# MAGIC
# MAGIC Para começar, vamos revisitar a visualização bidimensional dos nossos clusters para nos orientar sobre eles. A codificação de cores usada neste gráfico será aplicada em nossas visualizações restantes para facilitar a determinação do cluster em análise:

# COMMAND ----------

# DBTITLE 1,Visualize Clusters
# visualize cluster assignments
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
  data=labeled_finalfeatures_pd,
  x='Dim.1',
  y='Dim.2',
  hue=cluster_column,
  palette=cluster_colors,
  legend='brief',
  alpha=0.5,
  ax = ax
  )
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------

# MAGIC %md O design de segmento que criamos não produz grupos de tamanhos iguais. Em vez disso, temos um grupo um pouco maior do que os outros, embora os grupos menores ainda sejam de um tamanho útil para nossa equipe:

# COMMAND ----------

# DBTITLE 1,Count Cluster Members
# count members per cluster
cluster_member_counts = labeled_finalfeatures_pd.groupby([cluster_column]).agg({cluster_column:['count']})
cluster_member_counts.columns = cluster_member_counts.columns.droplevel(0)

# plot counts
plt.bar(
  cluster_member_counts.index,
  cluster_member_counts['count'],
  color = cluster_colors,
  tick_label=cluster_member_counts.index
  )

# stretch y-axis
plt.ylim(0,labeled_finalfeatures_pd.shape[0])

# labels
for index, value in zip(cluster_member_counts.index, cluster_member_counts['count']):
    plt.text(index, value, str(value)+'\n', horizontalalignment='center', verticalalignment='baseline')

# COMMAND ----------

# MAGIC %md Vamos agora examinar como cada segmento difere em relação às nossas características base. Para nossas características categóricas, iremos plotar a proporção de membros do cluster identificados como participantes de uma atividade promocional específica em relação ao número total de membros do cluster. Para nossas características contínuas, iremos visualizar os valores usando um gráfico de caixa:

# COMMAND ----------

# DBTITLE 1,Define Function to Render Plots
def profile_segments_by_features(data, features_to_plot, label_to_plot, label_count, label_colors):
  
    feature_count = len(features_to_plot)
    
    # configure plot layout
    max_cols = 5
    if feature_count > max_cols:
      column_count = max_cols
    else:
      column_count = feature_count      
      
    row_count = math.ceil(feature_count / column_count)

    fig, ax = plt.subplots(row_count, column_count, figsize =(column_count * 4, row_count * 4))
    
    # for each feature (enumerated)
    for k in range(feature_count):

      # determine row & col position
      col = k % column_count
      row = int(k / column_count)
      
      # get axis reference (can be 1- or 2-d)
      try:
        k_ax = ax[row,col]
      except:
        pass
        k_ax = ax[col]
      
      # set plot title
      k_ax.set_title(features_to_plot[k].replace('_',' '), fontsize=7)

      # CATEGORICAL FEATURES
      if features_to_plot[k][:4]=='has_': 

        # calculate members associated with 0/1 categorical values
        x = data.groupby([label_to_plot,features_to_plot[k]]).agg({label_to_plot:['count']})
        x.columns = x.columns.droplevel(0)

        # for each cluster
        for c in range(label_count):

          # get count of cluster members
          c_count = x.loc[c,:].sum()[0]

          # calculate members with value 0
          try:
            c_0 = x.loc[c,0]['count']/c_count
          except:
            c_0 = 0

          # calculate members with value 1
          try:
            c_1 = x.loc[c,1]['count']/c_count
          except:
            c_1 = 0

          # render percent stack bar chart with 1s on bottom and 0s on top
          k_ax.set_ylim(0,1)
          k_ax.bar([c], c_1, color=label_colors[c], edgecolor='white')
          k_ax.bar([c], c_0, bottom=c_1, color=label_colors[c], edgecolor='white', alpha=0.25)


      # CONTINUOUS FEATURES
      else:    

        # get subset of data with entries for this feature
        x = data[
              ~np.isnan(data[features_to_plot[k]])
              ][[label_to_plot,features_to_plot[k]]]

        # get values for each cluster
        p = []
        for c in range(label_count):
          p += [x[x[label_to_plot]==c][features_to_plot[k]].values]

        # plot values
        k_ax.set_ylim(0,1)
        bplot = k_ax.boxplot(
            p, 
            labels=range(label_count),
            patch_artist=True
            )

        # adjust box fill to align with cluster
        for patch, color in zip(bplot['boxes'], label_colors):
          patch.set_alpha(0.75)
          patch.set_edgecolor('black')
          patch.set_facecolor(color)
    

# COMMAND ----------

# DBTITLE 1,Render Plots for All Base Features
# get feature names
feature_names = labeled_basefeatures_pd.drop(label_columns, axis=1).columns

# generate plots
profile_segments_by_features(labeled_basefeatures_pd, feature_names, cluster_column, cluster_count, cluster_colors)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Há muito a examinar neste gráfico, mas a coisa mais fácil parece ser começar com as características categóricas para identificar grupos responsivos a algumas ofertas promocionais e não a outras. As características contínuas, então, fornecem um pouco mais de visão sobre o grau de engajamento quando esse grupo responde.
# MAGIC
# MAGIC À medida que você avança pelas várias características, provavelmente começará a formar descrições dos diferentes grupos. Para ajudar com isso, pode ser útil recuperar subconjuntos específicos de características para focar sua atenção em um número menor de características:

# COMMAND ----------

# DBTITLE 1,Plot Subset of Features
feature_names = ['has_pdates_campaign_targeted', 'pdates_campaign_targeted', 'amount_list_with_campaign_targeted']

profile_segments_by_features(labeled_basefeatures_pd, feature_names, cluster_column, cluster_count, cluster_colors)

# COMMAND ----------

# MAGIC %md ## Etapa 3: Descrever Segmentos
# MAGIC
# MAGIC Com uma análise cuidadosa das características, você deve, esperançosamente, conseguir diferenciar os clusters em termos de seu comportamento. Agora, torna-se interessante examinar por que esses grupos podem existir e/ou como poderíamos ser capazes de identificar a provável pertença a um grupo sem coletar vários anos de informações comportamentais. Uma maneira comum de fazer isso é examinar os clusters em termos de características que não foram empregadas no design do cluster. Com este conjunto de dados, podemos empregar informações demográficas disponíveis para um subconjunto de nossas famílias para esse propósito:

# COMMAND ----------

# DBTITLE 1,Associate Household Demographics with Cluster Labels
labels = spark.table('rodrigo_catalog.journey.household_clusters').alias('labels')
demographics = spark.table('rodrigo_catalog.journey.households').alias('demographics')

labeled_demos = (
  labels
    .join(demographics, on=expr('labels.household_id=demographics.household_id'), how='leftouter')  # only 801 of 2500 present should match
    .withColumn('matched', expr('demographics.household_id Is Not Null'))
    .drop('household_id')
  ).toPandas()

labeled_demos

# COMMAND ----------

# MAGIC %md Antes de prosseguir, precisamos considerar quantos dos nossos membros no cluster têm informações demográficas associadas a eles:

# COMMAND ----------

x = labeled_demos.groupby([cluster_column, 'matched']).agg({cluster_column:['count']}).reset_index()
x.columns = ['hc_cluster', 'matched', 'count']
display(x)

# COMMAND ----------

# DBTITLE 1,Examine Proportion of Cluster Members with Demographic Data
# for each cluster
for c in range(cluster_count):

  # get count of cluster members
  c_count = x.loc[c,:].sum()

  # calculate members with value 0
  try:
    c_0 = x.loc[(x['hc_cluster'] == c) & (x['matched'] == False), 'count']
  except:
    c_0 = 0

  # calculate members with value 1
  try:
    c_1 = x.loc[(x['hc_cluster'] == c) & (x['matched'] == True), 'count']
  except:
    c_1 = 0
  
  # plot counts
  plt.bar([c], c_1, color=cluster_colors[c], edgecolor='white')
  plt.bar([c], c_0, bottom=c_1, color=cluster_colors[c], edgecolor='white', alpha=0.25)
  plt.xticks(range(cluster_count))
  plt.ylim(0,1)

# COMMAND ----------

# MAGIC %md Idealmente, teríamos dados demográficos para todos os domicílios no conjunto de dados ou pelo menos para uma proporção grande e consistente de membros em cada cluster. Sem isso, precisamos ter cuidado ao tirar conclusões a partir desses dados.
# MAGIC
# MAGIC Ainda assim, podemos prosseguir com o exercício para demonstrar a técnica. Com isso em mente, vamos construir uma tabela de contingência para a faixa etária do chefe de família para ver como os membros do cluster se alinham em relação à idade:

# COMMAND ----------

# DBTITLE 1,Demonstrate Contingency Table
age_by_cluster = sm.stats.Table.from_data(labeled_demos[[cluster_column,'age_bracket']])
age_by_cluster.table_orig

# COMMAND ----------

# MAGIC %md Podemos então aplicar o teste Qui-quadrado de Pearson (*&Chi;^2*) para determinar se essas diferenças de frequência são estatisticamente significativas. Em um teste como esse, um valor de p igual ou menor que 5% nos diria que as distribuições de frequência não são prováveis devido ao acaso (e, portanto, dependem da atribuição de categoria):

# COMMAND ----------

# DBTITLE 1,Demonstrate Chi-Squared Test
res = age_by_cluster.test_nominal_association()
res.pvalue

# COMMAND ----------

# MAGIC %md Em seguida, poderíamos examinar os resíduos de Pearson associados à interseção de cada cluster e grupo demográfico para determinar quando interseções específicas estavam nos levando a essa conclusão. Interseções com valores residuais **absolutos** maiores que 2 ou 4 difeririam das expectativas com uma probabilidade de 95% ou 99,9%, respectivamente, e essas provavelmente seriam as características demográficas que diferenciariam os clusters:

# COMMAND ----------

# DBTITLE 1,Demonstrate Pearson Residuals
age_by_cluster.resid_pearson  # standard normal random variables within -2, 2 with 95% prob and -4,4 at 99.99% prob

# COMMAND ----------

# MAGIC %md Se tivéssemos encontrado algo significativo nesses dados, nosso próximo desafio seria comunicá-lo aos membros da equipe que não estão familiarizados com esses testes estatísticos. Uma maneira popular de fazer isso é por meio de um *[gráfico de mosaico](https://www.datavis.ca/papers/casm/casm.html#tth_sEc3)*, também conhecido como *gráfico de marimekko*:

# COMMAND ----------

# DBTITLE 1,Demonstrate Mosaic Plot
# assemble demographic category labels as key-value pairs (limit to matched values)
demo_labels = np.unique(labeled_demos[labeled_demos['matched']]['age_bracket'])
demo_labels_kv = dict(zip(demo_labels,demo_labels))

# define function to generate cell labels
labelizer = lambda key: demo_labels_kv[key[1]]

# define function to generate cell colors
props = lambda key: {'color': cluster_colors[int(key[0])], 'alpha':0.8}

# generate mosaic plot
fig, rect = mosaic(
  labeled_demos.sort_values('age_bracket', ascending=False),
  [cluster_column,'age_bracket'], 
  horizontal=True, 
  axes_label=True, 
  gap=0.015, 
  properties=props, 
  labelizer=labelizer
  )

# set figure size
_ = fig.set_size_inches((10,8))

# COMMAND ----------

# MAGIC %md A exibição proporcional dos membros associados a cada categoria, juntamente com a largura proporcional dos clusters em relação uns aos outros, fornece uma maneira interessante de resumir as diferenças de frequência entre esses grupos. Combinado com análise estatística, o gráfico de mosaico oferece uma maneira agradável de compreender mais facilmente uma descoberta estatisticamente significativa.

# COMMAND ----------

# MAGIC %md ## Passo 4: Próximos Passos
# MAGIC
# MAGIC A segmentação raramente é um exercício único. Em vez disso, aprendendo com essa análise, podemos repetir o processo, removendo características que não diferenciam e possivelmente incluindo outras. Além disso, podemos realizar outras análises, como segmentações RFM ou análise de CLV, e depois examinar como essas se relacionam com o design de segmentação explorado aqui. Eventualmente, podemos chegar a um novo design de segmentação, mas mesmo que não o façamos, adquirimos insights que podem nos ajudar a criar campanhas promocionais melhores.
