# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/segmentation.git. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/customer-segmentation.

# COMMAND ----------

# MAGIC %md O objetivo deste notebook é gerar as características necessárias para o nosso trabalho de segmentação usando uma combinação de técnicas de engenharia de características e redução de dimensão.

# COMMAND ----------

# DBTITLE 1,Install Required Python Libraries
# MAGIC %pip install dython==0.7.1

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from sklearn.preprocessing import quantile_transform

import dython
import math

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md ## Passo 1: Derivar Recursos Básicos
# MAGIC
# MAGIC Com o objetivo de segmentar os domicílios dos clientes com base em sua receptividade a vários esforços promocionais, começamos calculando o número de datas de compra (*pdates\_*) e o volume de vendas (*amount\_list_*) associados a cada item promocional, sozinho e em combinação com os outros. Os itens promocionais considerados são:
# MAGIC
# MAGIC * Produtos direcionados pela campanha (*campaign\_targeted_*)
# MAGIC * Produtos de marca própria (*private\_label_*)
# MAGIC * Produtos com desconto na loja (*instore\_discount_*)
# MAGIC * Resgates de cupons de campanha (gerados pelo varejista) (*campaign\_coupon\_redemption_*)
# MAGIC * Resgates de cupons gerados pelo fabricante (*manuf\_coupon\_redemption_*)
# MAGIC
# MAGIC As métricas resultantes não são exaustivas, mas fornecem um ponto de partida útil para nossa análise:

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC USE CATALOG rodrigo_catalog;
# MAGIC USE DATABASE journey;

# COMMAND ----------

# DBTITLE 1,Derive Relevant Metrics
# MAGIC %sql
# MAGIC
# MAGIC DROP VIEW IF EXISTS household_metrics;
# MAGIC
# MAGIC CREATE VIEW household_metrics
# MAGIC AS
# MAGIC   WITH 
# MAGIC     targeted_products_by_household AS (
# MAGIC       SELECT DISTINCT
# MAGIC         b.household_id,
# MAGIC         c.product_id
# MAGIC       FROM campaigns a
# MAGIC       INNER JOIN campaigns_households b
# MAGIC         ON a.campaign_id=b.campaign_id
# MAGIC       INNER JOIN coupons c
# MAGIC         ON a.campaign_id=c.campaign_id
# MAGIC       ),
# MAGIC     product_spend AS (
# MAGIC       SELECT
# MAGIC         a.household_id,
# MAGIC         a.product_id,
# MAGIC         a.day,
# MAGIC         a.basket_id,
# MAGIC         CASE WHEN a.campaign_coupon_discount > 0 THEN 1 ELSE 0 END as campaign_coupon_redemption,
# MAGIC         CASE WHEN a.manuf_coupon_discount > 0 THEN 1 ELSE 0 END as manuf_coupon_redemption,
# MAGIC         CASE WHEN a.instore_discount > 0 THEN 1 ELSE 0 END as instore_discount_applied,
# MAGIC         CASE WHEN b.brand = 'Private' THEN 1 ELSE 0 END as private_label,
# MAGIC         a.amount_list,
# MAGIC         a.campaign_coupon_discount,
# MAGIC         a.manuf_coupon_discount,
# MAGIC         a.total_coupon_discount,
# MAGIC         a.instore_discount,
# MAGIC         a.amount_paid  
# MAGIC       FROM transactions_adj a
# MAGIC       INNER JOIN products b
# MAGIC         ON a.product_id=b.product_id
# MAGIC       )
# MAGIC   SELECT
# MAGIC
# MAGIC     x.household_id,
# MAGIC
# MAGIC     -- Purchase Date Level Metrics
# MAGIC     COUNT(DISTINCT x.day) as purchase_dates,
# MAGIC     COUNT(DISTINCT CASE WHEN y.product_id IS NOT NULL THEN x.day ELSE NULL END) as pdates_campaign_targeted,
# MAGIC     COUNT(DISTINCT CASE WHEN x.private_label = 1 THEN x.day ELSE NULL END) as pdates_private_label,
# MAGIC     COUNT(DISTINCT CASE WHEN y.product_id IS NOT NULL AND x.private_label = 1 THEN x.day ELSE NULL END) as pdates_campaign_targeted_private_label,
# MAGIC     COUNT(DISTINCT CASE WHEN x.campaign_coupon_redemption = 1 THEN x.day ELSE NULL END) as pdates_campaign_coupon_redemptions,
# MAGIC     COUNT(DISTINCT CASE WHEN x.campaign_coupon_redemption = 1 AND x.private_label = 1 THEN x.day ELSE NULL END) as pdates_campaign_coupon_redemptions_on_private_labels,
# MAGIC     COUNT(DISTINCT CASE WHEN x.manuf_coupon_redemption = 1 THEN x.day ELSE NULL END) as pdates_manuf_coupon_redemptions,
# MAGIC     COUNT(DISTINCT CASE WHEN x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN y.product_id IS NOT NULL AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_campaign_targeted_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN x.private_label = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_private_label_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN y.product_id IS NOT NULL AND x.private_label = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_campaign_targeted_private_label_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN x.campaign_coupon_redemption = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_campaign_coupon_redemption_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN x.campaign_coupon_redemption = 1 AND x.private_label = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN x.manuf_coupon_redemption = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_manuf_coupon_redemption_instore_discount_applied,
# MAGIC
# MAGIC     -- List Amount Metrics
# MAGIC     COALESCE(SUM(x.amount_list),0) as amount_list,
# MAGIC     COALESCE(SUM(CASE WHEN y.product_id IS NOT NULL THEN 1 ELSE 0 END * x.amount_list),0) as amount_list_with_campaign_targeted,
# MAGIC     COALESCE(SUM(x.private_label * x.amount_list),0) as amount_list_with_private_label,
# MAGIC     COALESCE(SUM(CASE WHEN y.product_id IS NOT NULL THEN 1 ELSE 0 END * x.private_label * x.amount_list),0) as amount_list_with_campaign_targeted_private_label,
# MAGIC     COALESCE(SUM(x.campaign_coupon_redemption * x.amount_list),0) as amount_list_with_campaign_coupon_redemptions,
# MAGIC     COALESCE(SUM(x.campaign_coupon_redemption * x.private_label * x.amount_list),0) as amount_list_with_campaign_coupon_redemptions_on_private_labels,
# MAGIC     COALESCE(SUM(x.manuf_coupon_redemption * x.amount_list),0) as amount_list_with_manuf_coupon_redemptions,
# MAGIC     COALESCE(SUM(x.instore_discount_applied * x.amount_list),0) as amount_list_with_instore_discount_applied,
# MAGIC     COALESCE(SUM(CASE WHEN y.product_id IS NOT NULL THEN 1 ELSE 0 END * x.instore_discount_applied * x.amount_list),0) as amount_list_with_campaign_targeted_instore_discount_applied,
# MAGIC     COALESCE(SUM(x.private_label * x.instore_discount_applied * x.amount_list),0) as amount_list_with_private_label_instore_discount_applied,
# MAGIC     COALESCE(SUM(CASE WHEN y.product_id IS NOT NULL THEN 1 ELSE 0 END * x.private_label * x.instore_discount_applied * x.amount_list),0) as amount_list_with_campaign_targeted_private_label_instore_discount_applied,
# MAGIC     COALESCE(SUM(x.campaign_coupon_redemption * x.instore_discount_applied * x.amount_list),0) as amount_list_with_campaign_coupon_redemption_instore_discount_applied,
# MAGIC     COALESCE(SUM(x.campaign_coupon_redemption * x.private_label * x.instore_discount_applied * x.amount_list),0) as amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC     COALESCE(SUM(x.manuf_coupon_redemption * x.instore_discount_applied * x.amount_list),0) as amount_list_with_manuf_coupon_redemption_instore_discount_applied
# MAGIC
# MAGIC   FROM product_spend x
# MAGIC   LEFT OUTER JOIN targeted_products_by_household y
# MAGIC     ON x.household_id=y.household_id AND x.product_id=y.product_id
# MAGIC   GROUP BY 
# MAGIC     x.household_id;
# MAGIC     
# MAGIC SELECT * FROM household_metrics;

# COMMAND ----------

# MAGIC %md É assumido que os domicílios incluídos neste conjunto de dados foram selecionados com base em um nível mínimo de atividade ao longo do período de 711 dias em que os dados são fornecidos. No entanto, diferentes domicílios demonstram diferentes níveis de frequência de compra durante esse período, bem como diferentes níveis de gastos totais. Para normalizar esses valores entre os domicílios, dividiremos cada métrica pelo total de datas de compra ou pelo valor total da lista associado a esse domicílio ao longo de seu histórico de compras disponível:
# MAGIC
# MAGIC **NOTA** A normalização dos dados com base nas datas de compra totais e nos gastos, como fazemos nesta próxima etapa, pode não ser apropriada em todas as análises.

# COMMAND ----------

# DBTITLE 1,Convert Metrics to Features
# MAGIC %sql
# MAGIC
# MAGIC DROP VIEW IF EXISTS household_features;
# MAGIC
# MAGIC CREATE VIEW household_features 
# MAGIC AS 
# MAGIC
# MAGIC SELECT
# MAGIC       household_id,
# MAGIC   
# MAGIC       pdates_campaign_targeted/purchase_dates as pdates_campaign_targeted,
# MAGIC       pdates_private_label/purchase_dates as pdates_private_label,
# MAGIC       pdates_campaign_targeted_private_label/purchase_dates as pdates_campaign_targeted_private_label,
# MAGIC       pdates_campaign_coupon_redemptions/purchase_dates as pdates_campaign_coupon_redemptions,
# MAGIC       pdates_campaign_coupon_redemptions_on_private_labels/purchase_dates as pdates_campaign_coupon_redemptions_on_private_labels,
# MAGIC       pdates_manuf_coupon_redemptions/purchase_dates as pdates_manuf_coupon_redemptions,
# MAGIC       pdates_instore_discount_applied/purchase_dates as pdates_instore_discount_applied,
# MAGIC       pdates_campaign_targeted_instore_discount_applied/purchase_dates as pdates_campaign_targeted_instore_discount_applied,
# MAGIC       pdates_private_label_instore_discount_applied/purchase_dates as pdates_private_label_instore_discount_applied,
# MAGIC       pdates_campaign_targeted_private_label_instore_discount_applied/purchase_dates as pdates_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       pdates_campaign_coupon_redemption_instore_discount_applied/purchase_dates as pdates_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       pdates_campaign_coupon_redemption_private_label_instore_discount_applied/purchase_dates as pdates_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       pdates_manuf_coupon_redemption_instore_discount_applied/purchase_dates as pdates_manuf_coupon_redemption_instore_discount_applied,
# MAGIC       
# MAGIC       amount_list_with_campaign_targeted/amount_list as amount_list_with_campaign_targeted,
# MAGIC       amount_list_with_private_label/amount_list as amount_list_with_private_label,
# MAGIC       amount_list_with_campaign_targeted_private_label/amount_list as amount_list_with_campaign_targeted_private_label,
# MAGIC       amount_list_with_campaign_coupon_redemptions/amount_list as amount_list_with_campaign_coupon_redemptions,
# MAGIC       amount_list_with_campaign_coupon_redemptions_on_private_labels/amount_list as amount_list_with_campaign_coupon_redemptions_on_private_labels,
# MAGIC       amount_list_with_manuf_coupon_redemptions/amount_list as amount_list_with_manuf_coupon_redemptions,
# MAGIC       amount_list_with_instore_discount_applied/amount_list as amount_list_with_instore_discount_applied,
# MAGIC       amount_list_with_campaign_targeted_instore_discount_applied/amount_list as amount_list_with_campaign_targeted_instore_discount_applied,
# MAGIC       amount_list_with_private_label_instore_discount_applied/amount_list as amount_list_with_private_label_instore_discount_applied,
# MAGIC       amount_list_with_campaign_targeted_private_label_instore_discount_applied/amount_list as amount_list_with_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       amount_list_with_campaign_coupon_redemption_instore_discount_applied/amount_list as amount_list_with_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied/amount_list as amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       amount_list_with_manuf_coupon_redemption_instore_discount_applied/amount_list as amount_list_with_manuf_coupon_redemption_instore_discount_applied
# MAGIC
# MAGIC FROM household_metrics
# MAGIC ORDER BY household_id;
# MAGIC
# MAGIC SELECT * FROM household_features;

# COMMAND ----------

# MAGIC %md ## Passo 2: Examinar Distribuições
# MAGIC
# MAGIC Antes de prosseguir, é uma boa ideia examinar nossas características de perto para entender sua compatibilidade com as técnicas de agrupamento que podemos utilizar. Em geral, nossa preferência seria ter dados padronizados com distribuições relativamente normais, embora isso não seja um requisito rígido para todos os algoritmos de agrupamento.
# MAGIC
# MAGIC Para nos ajudar a examinar as distribuições dos dados, vamos extrair nossos dados para um pandas Dataframe. Se o volume de dados fosse muito grande para o pandas, poderíamos considerar a obtenção de uma amostra aleatória (usando o [*sample()*](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.sample) no DataFrame do Spark) para examinar as distribuições:

# COMMAND ----------

# DBTITLE 1,Retrieve Features
# retrieve as Spark dataframe
household_features = (
  spark
    .table('household_features')
  )

# retrieve as pandas Dataframe
household_features_pd = household_features.toPandas()

# collect some basic info on our features
household_features_pd.info()

# COMMAND ----------

# MAGIC %md Observe que optamos por recuperar o campo *household_id* com este conjunto de dados. Identificadores únicos como este não serão passados para a transformação de dados e o trabalho de agrupamento que segue, mas podem ser úteis para nos ajudar a validar os resultados desse trabalho. Ao recuperar essas informações com nossas características, agora podemos separar nossas características e o identificador único em dois dataframes separados do pandas, onde as instâncias em cada um podem ser facilmente reassociadas usando um valor de índice compartilhado:

# COMMAND ----------

# DBTITLE 1,Separate Household ID from Features
# get household ids from dataframe
households_pd = household_features_pd[['household_id']]

# remove household ids from dataframe
features_pd = household_features_pd.drop(['household_id'], axis=1)

features_pd

# COMMAND ----------

# MAGIC %md Vamos agora começar a examinar a estrutura das nossas características:

# COMMAND ----------

# DBTITLE 1,Summary Stats on Features
features_pd.describe()

# COMMAND ----------

# MAGIC %md Uma rápida revisão das características mostra que muitas têm médias muito baixas e um grande número de valores zero (como indicado pela ocorrência de zeros em várias posições de quantil). Devemos dar uma olhada mais de perto na distribuição de nossas características para garantir que não tenhamos problemas de distribuição de dados que possam nos atrapalhar posteriormente:

# COMMAND ----------

# DBTITLE 1,Examine Feature Distributions
feature_names = features_pd.columns
feature_count = len(feature_names)

# determine required rows and columns for visualizations
column_count = 5
row_count = math.ceil(feature_count / column_count)

# configure figure layout
fig, ax = plt.subplots(row_count, column_count, figsize =(column_count * 4.5, row_count * 3))

# render distribution of each feature
for k in range(0,feature_count):
  
  # determine row & col position
  col = k % column_count
  row = int(k / column_count)
  
  # set figure at row & col position
  ax[row][col].hist(features_pd[feature_names[k]], rwidth=0.95, bins=10) # histogram
  ax[row][col].set_xlim(0,1)   # set x scale 0 to 1
  ax[row][col].set_ylim(0,features_pd.shape[0]) # set y scale 0 to 2500 (household count)
  ax[row][col].text(x=0.1, y=features_pd.shape[0]-100, s=feature_names[k].replace('_','\n'), fontsize=9, va='top')      # feature name in chart

# COMMAND ----------

# MAGIC %md Uma rápida inspeção visual mostra que temos distribuições com excesso de zeros associadas a muitas de nossas características. Isso é um desafio comum quando uma característica tenta medir a magnitude de um evento que ocorre com baixa frequência.
# MAGIC
# MAGIC Existe uma quantidade crescente de literatura descrevendo várias técnicas para lidar com distribuições com excesso de zeros e até mesmo alguns modelos com excesso de zeros projetados para trabalhar com eles. Para nossos propósitos, simplesmente separaremos as características com essas distribuições em duas características, uma das quais capturará a ocorrência do evento como uma característica binária (categórica) e a outra capturará a magnitude do evento quando ocorrer:
# MAGIC
# MAGIC **NOTA** Rotularemos nossas características binárias com o prefixo *has\_* para torná-las mais facilmente identificáveis. Esperamos que se uma família tiver zero datas de compra associadas a um evento, também esperamos que essa família não tenha valores de venda para esse evento. Com isso em mente, criaremos uma única característica binária para um evento e uma característica secundária para cada uma das datas de compra e valores da lista de quantidades associadas.

# COMMAND ----------

# DBTITLE 1,Define Features to Address Zero-Inflated Distribution Problem
# MAGIC %sql
# MAGIC
# MAGIC DROP VIEW IF EXISTS household_features;
# MAGIC
# MAGIC CREATE VIEW household_features 
# MAGIC AS 
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC       household_id,
# MAGIC       
# MAGIC       -- binary features
# MAGIC       CASE WHEN pdates_campaign_targeted > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_targeted,
# MAGIC       -- CASE WHEN pdates_private_label > 0 THEN 1 ELSE 0 END as has_pdates_private_label,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_targeted_private_label,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_coupon_redemptions,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions_on_private_labels > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_coupon_redemptions_on_private_labels,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemptions > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_manuf_coupon_redemptions,
# MAGIC       -- CASE WHEN pdates_instore_discount_applied > 0 THEN 1 ELSE 0 END as has_pdates_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_targeted_instore_discount_applied,
# MAGIC       -- CASE WHEN pdates_private_label_instore_discount_applied > 0 THEN 1 ELSE 0 END as has_pdates_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_private_label_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemption_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_manuf_coupon_redemption_instore_discount_applied,
# MAGIC   
# MAGIC       -- purchase date features
# MAGIC       CASE WHEN pdates_campaign_targeted > 0 THEN pdates_campaign_targeted/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_targeted,
# MAGIC       pdates_private_label/purchase_dates as pdates_private_label,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label > 0 THEN pdates_campaign_targeted_private_label/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_targeted_private_label,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions > 0 THEN pdates_campaign_coupon_redemptions/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_coupon_redemptions,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions_on_private_labels > 0 THEN pdates_campaign_coupon_redemptions_on_private_labels/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_coupon_redemptions_on_private_labels,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemptions > 0 THEN pdates_manuf_coupon_redemptions/purchase_dates 
# MAGIC         ELSE NULL END as pdates_manuf_coupon_redemptions,
# MAGIC       CASE WHEN pdates_campaign_targeted_instore_discount_applied > 0 THEN pdates_campaign_targeted_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_targeted_instore_discount_applied,
# MAGIC       pdates_private_label_instore_discount_applied/purchase_dates as pdates_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label_instore_discount_applied > 0 THEN pdates_campaign_targeted_private_label_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_instore_discount_applied > 0 THEN pdates_campaign_coupon_redemption_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_private_label_instore_discount_applied > 0 THEN pdates_campaign_coupon_redemption_private_label_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemption_instore_discount_applied > 0 THEN pdates_manuf_coupon_redemption_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_manuf_coupon_redemption_instore_discount_applied,
# MAGIC       
# MAGIC       -- list amount features
# MAGIC       CASE WHEN pdates_campaign_targeted > 0 THEN amount_list_with_campaign_targeted/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_targeted,
# MAGIC       amount_list_with_private_label/amount_list as amount_list_with_private_label,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label > 0 THEN amount_list_with_campaign_targeted_private_label/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_targeted_private_label,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions > 0 THEN amount_list_with_campaign_coupon_redemptions/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_coupon_redemptions,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions_on_private_labels > 0 THEN amount_list_with_campaign_coupon_redemptions_on_private_labels/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_coupon_redemptions_on_private_labels,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemptions > 0 THEN amount_list_with_manuf_coupon_redemptions/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_manuf_coupon_redemptions,
# MAGIC       amount_list_with_instore_discount_applied/amount_list as amount_list_with_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_instore_discount_applied > 0 THEN amount_list_with_campaign_targeted_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_targeted_instore_discount_applied,
# MAGIC       amount_list_with_private_label_instore_discount_applied/amount_list as amount_list_with_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label_instore_discount_applied > 0 THEN amount_list_with_campaign_targeted_private_label_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_instore_discount_applied > 0 THEN amount_list_with_campaign_coupon_redemption_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_private_label_instore_discount_applied > 0 THEN amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemption_instore_discount_applied > 0 THEN amount_list_with_manuf_coupon_redemption_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_manuf_coupon_redemption_instore_discount_applied
# MAGIC
# MAGIC FROM household_metrics
# MAGIC ORDER BY household_id;

# COMMAND ----------

# DBTITLE 1,Read Features to Pandas
# retrieve as Spark dataframe
household_features = (
  spark
    .table('household_features')
  )

# retrieve as pandas Dataframe
household_features_pd = household_features.toPandas()

# get household ids from dataframe
households_pd = household_features_pd[['household_id']]

# remove household ids from dataframe
features_pd = household_features_pd.drop(['household_id'], axis=1)

features_pd

# COMMAND ----------

# MAGIC %md Com nossas características separadas, vamos olhar novamente as distribuições das nossas características. Vamos começar examinando nossas novas características binárias:

# COMMAND ----------

# DBTITLE 1,Examine Distribution of Binary Features
b_feature_names = list(filter(lambda f:f[0:4]==('has_') , features_pd.columns))
b_feature_count = len(b_feature_names)

# determine required rows and columns
b_column_count = 5
b_row_count = math.ceil(b_feature_count / b_column_count)

# configure figure layout
fig, ax = plt.subplots(b_row_count, b_column_count, figsize =(b_column_count * 3.5, b_row_count * 3.5))

# render distribution of each feature
for k in range(0,b_feature_count):
  
  # determine row & col position
  b_col = k % b_column_count
  b_row = int(k / b_column_count)
  
  # determine feature to be plotted
  f = b_feature_names[k]
  
  value_counts = features_pd[f].value_counts()

  # render pie chart
  ax[b_row][b_col].pie(
    x = value_counts.values,
    labels = value_counts.index,
    explode = None,
    autopct='%1.1f%%',
    labeldistance=None,
    #pctdistance=0.4,
    frame=True,
    radius=0.48,
    center=(0.5, 0.5)
    )
  
  # clear frame of ticks
  ax[b_row][b_col].set_xticks([])
  ax[b_row][b_col].set_yticks([])
  
  # legend & feature name
  ax[b_row][b_col].legend(bbox_to_anchor=(1.04,1.05),loc='upper left', fontsize=8)
  ax[b_row][b_col].text(1.04,0.8, s=b_feature_names[k].replace('_','\n'), fontsize=8, va='top')

# COMMAND ----------

# MAGIC %md Pelos gráficos de pizza, parece que muitas ofertas promocionais não são aproveitadas. Isso é típico para a maioria das ofertas promocionais, especialmente aquelas associadas a cupons. Individualmente, vemos baixa adesão a muitas ofertas promocionais, mas quando examinamos a adesão de várias ofertas promocionais em combinação umas com as outras, a frequência de adesão cai para níveis em que podemos considerar ignorar as ofertas em combinação, em vez de focar nelas individualmente. Vamos adiar a abordagem desse assunto para voltar nossa atenção para nossas características contínuas, muitas das quais agora estão corrigidas para a inflação de zeros:

# COMMAND ----------

# DBTITLE 1,Examine Distribution of Continuous Features
c_feature_names = list(filter(lambda f:f[0:4]!=('has_') , features_pd.columns))
c_feature_count = len(c_feature_names)

# determine required rows and columns
c_column_count = 5
c_row_count = math.ceil(c_feature_count / c_column_count)

# configure figure layout
fig, ax = plt.subplots(c_row_count, c_column_count, figsize =(c_column_count * 4.5, c_row_count * 3))

# render distribution of each feature
for k in range(0, c_feature_count):
  
  # determine row & col position
  c_col = k % c_column_count
  c_row = int(k / c_column_count)
  
  # determine feature to be plotted
  f = c_feature_names[k]
  
  # set figure at row & col position
  ax[c_row][c_col].hist(features_pd[c_feature_names[k]], rwidth=0.95, bins=10) # histogram
  ax[c_row][c_col].set_xlim(0,1)   # set x scale 0 to 1
  ax[c_row][c_col].set_ylim(0,features_pd.shape[0]) # set y scale 0 to 2500 (household count)
  ax[c_row][c_col].text(x=0.1, y=features_pd.shape[0]-100, s=c_feature_names[k].replace('_','\n'), fontsize=9, va='top')      # feature name in chart

# COMMAND ----------

# MAGIC %md Com os zeros removidos de muitas de nossas características problemáticas, agora temos distribuições mais padronizadas. No entanto, muitas dessas distribuições não são normais (não gaussianas), e distribuições gaussianas podem ser realmente úteis com muitas técnicas de agrupamento.
# MAGIC
# MAGIC Uma maneira de tornar essas distribuições mais normais é aplicar a transformação de Box-Cox. Em nossa aplicação dessa transformação a essas características (não mostrada), descobrimos que muitas das distribuições não se tornaram muito mais normais do que o mostrado aqui. Portanto, vamos usar outra transformação que é um pouco mais assertiva, a [transformação de quantil](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.quantile_transform.html#sklearn.preprocessing.quantile_transform).
# MAGIC
# MAGIC A transformação de quantil calcula a função de probabilidade cumulativa associada aos pontos de dados de uma determinada característica. Isso é uma maneira sofisticada de dizer que os dados de uma característica são ordenados e uma função para calcular a classificação percentual de um valor dentro do intervalo de valores observados é calculada. Essa função de classificação percentual fornece a base para mapear os dados para uma distribuição conhecida, como uma distribuição normal. A [matemática exata](https://www.sciencedirect.com/science/article/abs/pii/S1385725853500125) por trás dessa transformação não precisa ser totalmente compreendida para que a utilidade dessa transformação seja observada. Se esta é sua primeira introdução às transformações de quantil, saiba que a técnica existe desde a década de 1950 e é amplamente utilizada em muitas disciplinas acadêmicas:

# COMMAND ----------

# DBTITLE 1,Apply Quantile Transformation to Continuous Features
# access continuous features
c_features_pd = features_pd[c_feature_names]

# apply quantile transform
qc_features_pd = pd.DataFrame(
  quantile_transform(c_features_pd, output_distribution='normal', ignore_implicit_zeros=True),
  columns=c_features_pd.columns,
  copy=True
  )

# show transformed data
qc_features_pd

# COMMAND ----------

# DBTITLE 1,Examine Distribution of Quantile-Transformed Continuous Features
qc_feature_names = qc_features_pd.columns
qc_feature_count = len(qc_feature_names)

# determine required rows and columns
qc_column_count = 5
qc_row_count = math.ceil(qc_feature_count / qc_column_count)

# configure figure layout
fig, ax = plt.subplots(qc_row_count, qc_column_count, figsize =(qc_column_count * 5, qc_row_count * 4))

# render distribution of each feature
for k in range(0,qc_feature_count):
  
  # determine row & col position
  qc_col = k % qc_column_count
  qc_row = int(k / qc_column_count)
  
  # set figure at row & col position
  ax[qc_row][qc_col].hist(qc_features_pd[qc_feature_names[k]], rwidth=0.95, bins=10) # histogram
  #ax[qc_row][qc_col].set_xlim(0,1)   # set x scale 0 to 1
  ax[qc_row][qc_col].set_ylim(0,features_pd.shape[0]) # set y scale 0 to 2500 (household count)
  ax[qc_row][qc_col].text(x=0.1, y=features_pd.shape[0]-100, s=qc_feature_names[k].replace('_','\n'), fontsize=9, va='top')      # feature name in chart

# COMMAND ----------

# MAGIC %md É importante notar que, por mais poderosa que seja a transformação quantílica, ela não resolve magicamente todos os problemas de dados. Ao desenvolver este notebook, identificamos várias características após a transformação em que parecia haver uma distribuição bimodal dos dados. Essas características eram aquelas para as quais inicialmente decidimos não aplicar a correção de distribuição com inflação de zeros. Ao retornar às nossas definições de características, implementar a correção e executar novamente a transformação resolveu o problema para nós. Dito isso, não corrigimos todas as distribuições transformadas em que há um pequeno grupo de domicílios posicionados à extrema esquerda da distribuição. Decidimos abordar apenas aqueles em que cerca de 250+ domicílios caíram nessa faixa.

# COMMAND ----------

# MAGIC %md ## Passo 3: Examinar Relacionamentos
# MAGIC
# MAGIC Agora que temos nossas características contínuas alinhadas com uma distribuição normal, vamos examinar a relação entre nossas variáveis de características, começando com nossas características contínuas. Usando a correlação padrão, podemos ver que temos um grande número de características altamente relacionadas. A multicolinearidade capturada aqui, se não for tratada, fará com que nosso agrupamento enfatize demais alguns aspectos da resposta à promoção em detrimento de outros:

# COMMAND ----------

# DBTITLE 1,Examine Relationships between Continuous Features
# generate correlations between features
qc_features_corr = qc_features_pd.corr()

# assemble a mask to remove top-half of heatmap
top_mask = np.zeros(qc_features_corr.shape, dtype=bool)
top_mask[np.triu_indices(len(top_mask))] = True

# define size of heatmap (for large number of features)
plt.figure(figsize=(10,8))

# generate heatmap
hmap = sns.heatmap(
  qc_features_corr,
  cmap = 'coolwarm',
  vmin =  1.0, 
  vmax = -1.0,
  mask = top_mask
  )

# COMMAND ----------

# MAGIC %md E quanto às relações entre nossas características binárias? A correlação de Pearson (usada no mapa de calor acima) não produz resultados válidos ao lidar com dados categóricos. Portanto, em vez disso, calcularemos o [Coeficiente de Incerteza de Theil](https://en.wikipedia.org/wiki/Uncertainty_coefficient), uma métrica projetada para examinar em que medida o valor de uma medida binária prevê outra. O U de Theil varia entre 0, onde não há valor preditivo entre as variáveis, e 1, onde há um valor preditivo perfeito. O interessante dessa métrica é que ela é **assimétrica**, ou seja, o escore mostra como uma medida binária prevê a outra, mas não necessariamente o contrário. Isso significa que precisamos examinar cuidadosamente os escores no mapa de calor abaixo e não assumir uma simetria na saída ao redor da diagonal:
# MAGIC
# MAGIC **NOTA** O autor principal do pacote *dython* do qual estamos usando o cálculo da métrica tem [um excelente artigo](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9) discutindo o U de Theil e métricas relacionadas.

# COMMAND ----------

# DBTITLE 1,Examine Relationships between Binary Features
from pyspark.sql.functions import regexp_replace, col
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Convert Pandas DataFrame to Spark DataFrame
features_df = spark.createDataFrame(features_pd)

# Clean the column by removing non-numeric values or replacing them with appropriate numeric values
# For example, you can remove non-numeric values using regular expressions
for feature_name in b_feature_names:
    features_df = features_df.withColumn(feature_name, regexp_replace(col(feature_name), '[^0-9.]', ''))

# Convert the data type to float
for feature_name in b_feature_names:
    features_df = features_df.withColumn(feature_name, features_df[feature_name].cast('float'))

# Convert Spark DataFrame to Pandas DataFrame
features_pd = features_df.toPandas()

# Generate heatmap with Theil's U
corr = features_pd[b_feature_names].corr(method='spearman')
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm', vmax=1.0, vmin=0.0, cbar=False)
plt.show()

# COMMAND ----------

# MAGIC %md Assim como com nossas características contínuas, temos algumas relações problemáticas entre nossas variáveis binárias que precisamos abordar. E quanto à relação entre as características contínuas e categóricas?
# MAGIC
# MAGIC Sabemos, a partir de como elas foram derivadas, que uma característica binária com um valor de 0 terá um valor NULL/NaN para suas características contínuas relacionadas e que qualquer valor real para uma característica contínua se traduzirá em um valor de 1 para a característica binária associada. Não precisamos calcular uma métrica para saber que temos uma relação entre essas características (embora o cálculo de uma [Razão de Correlação](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9) possa nos ajudar se tivermos alguma dúvida). Então, o que faremos para lidar com essas relações e as relações mencionadas anteriormente em nossos dados de características?
# MAGIC
# MAGIC Ao lidar com um grande número de características, essas relações são normalmente abordadas usando técnicas de redução de dimensionalidade. Essas técnicas projetam os dados de tal forma que a maior parte da variação nos dados é capturada por um número menor de características. Essas características, frequentemente referidas como fatores latentes ou componentes principais (dependendo da técnica empregada), capturam a estrutura subjacente dos dados que é refletida nas características de nível superficial, e o fazem de uma maneira que o poder explicativo sobreposto das características, ou seja, a multicolinearidade, é removido.
# MAGIC
# MAGIC Então, qual técnica de redução de dimensionalidade devemos usar? **Análise de Componentes Principais (PCA)** é a mais popular dessas técnicas, mas só pode ser aplicada a conjuntos de dados compostos por características contínuas. **Análise de Componentes Mistas (MCA)** é outra dessas técnicas, mas só pode ser aplicada a conjuntos de dados com características categóricas. **Análise de Fatores de Dados Mistos (FAMD)** nos permite combinar conceitos dessas duas técnicas para construir um conjunto de características reduzido quando nossos dados consistem em dados contínuos e categóricos. Dito isso, temos um problema ao aplicar FAMD aos nossos dados de características.
# MAGIC
# MAGIC Implementações típicas tanto de PCA quanto de MCA (e, portanto, FAMD) exigem que não haja valores de dados ausentes no conjunto de dados. A simples imputação usando valores médios ou medianos para características contínuas e valores frequentemente ocorrentes para características categóricas não funcionará, pois as técnicas de redução de dimensionalidade se baseiam na variação no conjunto de dados, e essas imputações simplesmente alteram fundamentalmente essa variação. (Para mais informações sobre isso, confira [este excelente vídeo](https://www.youtube.com/watch?v=OOM8_FH6_8o&feature=youtu.be). O vídeo está focado em PCA, mas as informações fornecidas são aplicáveis a todas essas técnicas.)
# MAGIC
# MAGIC Para imputar os dados corretamente, precisamos examinar a distribuição dos dados existentes e aproveitar as relações entre as características para imputar valores apropriados dessa distribuição de forma que não altere as projeções. O trabalho nesse espaço é bastante recente, mas alguns estatísticos desenvolveram a mecânica não apenas para PCA e MCA, mas também para FAMD. Nosso desafio é que não existem bibliotecas que implementem essas técnicas em Python, mas existem pacotes para isso em R.
# MAGIC
# MAGIC Agora precisamos transferir nossos dados para o R. Para fazer isso, vamos criar uma visualização temporária dos nossos dados com o mecanismo Spark SQL. Isso nos permitirá consultar esses dados a partir do R:

# COMMAND ----------

# DBTITLE 1,Register Transformed Data as Spark DataFrame
# assemble full dataset with transformed features
trans_features_pd = pd.concat([ 
  households_pd,  # add household IDs as supplemental variable
  qc_features_pd, 
  features_pd[b_feature_names].astype(str)
  ], axis=1)

# send dataset to spark as temp table
spark.createDataFrame(trans_features_pd).createOrReplaceTempView('trans_features_pd')

# COMMAND ----------

# MAGIC %md Agora iremos preparar nosso ambiente R carregando os pacotes necessários para o nosso trabalho. O pacote [FactoMineR](https://www.rdocumentation.org/packages/FactoMineR/versions/2.4) nos fornece a funcionalidade FAMD necessária, enquanto o pacote [missMDA](https://www.rdocumentation.org/packages/missMDA/versions/1.18) nos fornece capacidades de imputação:

# COMMAND ----------

# DBTITLE 1,Install Required R Packages
# MAGIC %r
# MAGIC require(devtools)
# MAGIC install.packages( c( "pbkrtest", "FactoMineR", "missMDA", "factoextra"), repos = "https://packagemanager.posit.co/cran/2022-09-08")

# COMMAND ----------

# MAGIC %md Agora podemos trazer nossos dados para o R. Observe que recuperamos os dados para um SparkR DataFrame antes de coletá-los em um data frame R local:

# COMMAND ----------

# DBTITLE 1,Retrieve Spark Data to R Data Frame
# MAGIC %r
# MAGIC
# MAGIC # retrieve data from from Spark
# MAGIC library(SparkR)
# MAGIC df.spark <- SparkR::sql("SELECT * FROM trans_features_pd")
# MAGIC
# MAGIC # move data to R data frame
# MAGIC df.r <- SparkR::collect(df.spark)
# MAGIC
# MAGIC summary(df.r)

# COMMAND ----------

# MAGIC %md Parece que os dados foram transferidos corretamente, mas precisamos examinar como as características binárias foram traduzidas. O FactoMiner e o missMDA exigem que as características categóricas sejam identificadas como tipos [*factor*](https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/factor) e aqui podemos ver que elas estão sendo apresentadas como caracteres:

# COMMAND ----------

# DBTITLE 1,Examine the R Data Frame's Structure
# MAGIC %r
# MAGIC
# MAGIC str(df.r)

# COMMAND ----------

# MAGIC %md Para converter nossas características categóricas em fatores, aplicamos uma conversão rápida:

# COMMAND ----------

# DBTITLE 1,Convert Categorical Features to Factors
# MAGIC %r
# MAGIC library(dplyr)
# MAGIC df.mutated <- mutate_if(df.r, is.character, as.factor)
# MAGIC
# MAGIC str(df.mutated)

# COMMAND ----------

# MAGIC %md Agora que os dados estão estruturados da maneira correta para nossa análise, podemos começar o trabalho de realizar o FAMD. Nosso primeiro passo é determinar o número de componentes principais necessários. O pacote missMDA fornece o método *estim_ncpFAMD* para esse propósito, mas observe que esse procedimento **demora muito para ser concluído**. Incluímos o código que usamos para executá-lo, mas o comentamos e substituímos pelo resultado que ele eventualmente encontrou durante nossa execução:

# COMMAND ----------

# MAGIC %r
# MAGIC install.packages("devtools")
# MAGIC devtools::install_version("Rserve")

# COMMAND ----------

# MAGIC %r
# MAGIC install.packages("missMDA")

# COMMAND ----------

# DBTITLE 1,Determine Number of Components
# MAGIC %r
# MAGIC
# MAGIC library(missMDA)
# MAGIC
# MAGIC # determine number of components to produce
# MAGIC #nb <- estim_ncpFAMD(df.mutated, ncp.max=10, sup.var=1)
# MAGIC nb <- list( c(8) ) 
# MAGIC names(nb) <- c("ncp")
# MAGIC
# MAGIC # display optimal number of components
# MAGIC nb$ncp

# COMMAND ----------

# MAGIC %md Com o número de componentes principais determinado, agora podemos imputar os valores ausentes. Por favor, observe que o FAMD, assim como o PCA e o MCA, requer que as variáveis sejam padronizadas. Os mecanismos para isso diferem dependendo se a variável é contínua ou categórica. O método *imputeFAMD* fornece funcionalidade para lidar com isso, com a configuração apropriada do argumento *scale*:

# COMMAND ----------

# DBTITLE 1,Impute Missing Values & Perform FAMD Transformation
# MAGIC %r 
# MAGIC
# MAGIC # impute missing values
# MAGIC library(missMDA)
# MAGIC
# MAGIC res.impute <- imputeFAMD(
# MAGIC   df.mutated,      # dataset with categoricals organized as factors
# MAGIC   ncp=nb$ncp,      # number of principal components
# MAGIC   scale=True,      # standardize features
# MAGIC   max.iter=10000,  # iterations to find optimal solution
# MAGIC   sup.var=1        # ignore the household_id field (column 1)
# MAGIC   ) 
# MAGIC
# MAGIC # perform FAMD
# MAGIC library(FactoMineR)
# MAGIC
# MAGIC res.famd <- FAMD(
# MAGIC   df.mutated,     # dataset with categoricals organized as factors
# MAGIC   ncp=nb$ncp,     # number of principal components
# MAGIC   tab.disj=res.impute$tab.disj, # imputation matrix from prior step
# MAGIC   sup.var=1,       # ignore the household_id field (column 1)
# MAGIC   graph=FALSE
# MAGIC )

# COMMAND ----------

# MAGIC %md Cada componente principal gerado pelo FAMD representa uma porcentagem da variância encontrada no conjunto de dados geral. A porcentagem de cada componente principal, identificado como dimensões 1 a 8, é capturada na saída do FAMD juntamente com a variância acumulada pelos componentes principais:

# COMMAND ----------

# DBTITLE 1,Plot Variance Captured by Components
# MAGIC %r
# MAGIC if (!requireNamespace("factoextra", quietly = TRUE)) {
# MAGIC   install.packages("factoextra")
# MAGIC }
# MAGIC
# MAGIC library("ggplot2")
# MAGIC library("factoextra")
# MAGIC
# MAGIC eig.val <- get_eigenvalue(res.famd)
# MAGIC print(eig.val)

# COMMAND ----------

# MAGIC %md Analisando essa saída, podemos ver que as duas primeiras dimensões (componentes principais) representam cerca de 50% da variância, permitindo-nos ter uma ideia da estrutura dos nossos dados por meio de uma visualização em 2-D:

# COMMAND ----------

# DBTITLE 1,Visualize Households Leveraging First Two Components
# MAGIC %r
# MAGIC
# MAGIC fviz_famd_ind(
# MAGIC   res.famd, 
# MAGIC   axes=c(1,2),  # use principal components 1 & 2
# MAGIC   geom = "point",  # show just the points (households)
# MAGIC   col.ind = "cos2", # color points (roughly) by the degree to which the principal component predicts the instance
# MAGIC   gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
# MAGIC   alpha.ind=0.5
# MAGIC   )

# COMMAND ----------

# MAGIC %md Ao plotar nossos domicílios pelos primeiros e segundos componentes principais, podemos observar que pode haver alguns agrupamentos interessantes de domicílios nos dados (conforme indicado pelos padrões de agrupamento no gráfico). Em um nível mais alto, nossos dados podem indicar alguns grandes agrupamentos bem separados, enquanto em um nível mais baixo, pode haver alguns agrupamentos mais refinados com fronteiras sobrepostas dentro dos agrupamentos maiores.
# MAGIC
# MAGIC Existem [muitos outros tipos de visualizações e análises que podemos realizar](http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/115-famd-factor-analysis-of-mixed-data-in-r-essentials/) nos resultados do FAMD para obter uma melhor compreensão de como nossas características base são representadas em cada um dos componentes principais, mas temos o que precisamos para fins de agrupamento. Agora vamos nos concentrar em obter os dados do R de volta para o Python.
# MAGIC
# MAGIC Para começar, vamos recuperar os valores dos componentes principais para cada um dos nossos domicílios:

# COMMAND ----------

# DBTITLE 1,Retrieve Household-Specific Values for Principal Components (Eigenvalues)
# MAGIC %r
# MAGIC
# MAGIC df.famd <- bind_cols(
# MAGIC   dplyr::select(df.r, "household_id"), 
# MAGIC   as.data.frame( res.famd$ind$coord ) 
# MAGIC   )
# MAGIC
# MAGIC head(df.famd)

# COMMAND ----------

# MAGIC %r
# MAGIC
# MAGIC write.csv(df.famd, "df_famd.csv", row.names = FALSE)

# COMMAND ----------

import pandas as pd

df_famd = pd.read_csv('df_famd.csv')
df_famd.head()

# COMMAND ----------

# DBTITLE 1,Persist Eigenvalues to Delta
# Convert df_famd to Spark DataFrame
df_famd_spark = spark.createDataFrame(df_famd)

# Write df_famd_spark to the 'rodrigo_catalog.journey.features_finalized' table
df_famd_spark.write.format("delta").mode("overwrite").saveAsTable("rodrigo_catalog.journey.features_finalized")

# COMMAND ----------

# DBTITLE 1,Retrieve Eigenvalues in Python
display(
  spark.table('rodrigo_catalog.journey.features_finalized')
  )

# COMMAND ----------

# MAGIC %md E agora vamos examinar as relações entre essas características:

# COMMAND ----------

# DBTITLE 1,Examine Relationships between Reduced Dimensions
# generate correlations between features
famd_features_corr = spark.table('rodrigo_catalog.journey.features_finalized').drop('household_id').toPandas().corr()

# assemble a mask to remove top-half of heatmap
top_mask = np.zeros(famd_features_corr.shape, dtype=bool)
top_mask[np.triu_indices(len(top_mask))] = True

# define size of heatmap (for large number of features)
plt.figure(figsize=(10,8))

# generate heatmap
hmap = sns.heatmap(
  famd_features_corr,
  cmap = 'coolwarm',
  vmin =  1.0, 
  vmax = -1.0,
  mask = top_mask
  )

# COMMAND ----------

# MAGIC %md Com a multicolinearidade abordada por meio do nosso conjunto de recursos reduzido, agora podemos prosseguir com a clusterização.
