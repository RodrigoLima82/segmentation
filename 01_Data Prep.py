# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/segmentation.git. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/customer-segmentation.

# COMMAND ----------

# MAGIC %md O objetivo deste notebook é acessar e preparar os dados necessários para o nosso trabalho de segmentação.

# COMMAND ----------

# MAGIC %md ## Passo 1: Acessar os Dados
# MAGIC
# MAGIC O objetivo deste acelerador é demonstrar como uma equipe de Gerenciamento de Promoções interessada em segmentar os domicílios dos clientes com base na responsividade às promoções pode realizar a análise. O conjunto de dados que usaremos foi disponibilizado pela Dunnhumby por meio do Kaggle e é chamado de [*The Complete Journey*](https://www.kaggle.com/frtgnn/dunnhumby-the-complete-journey). Ele consiste em vários arquivos que identificam a atividade de compra dos domicílios em combinação com várias campanhas promocionais para cerca de 2.500 domicílios ao longo de quase 2 anos. O esquema do conjunto de dados geral pode ser representado da seguinte forma:
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/segmentation_journey_schema3.png' width=500>

# COMMAND ----------

# MAGIC %run "./config/Data Extract"

# COMMAND ----------

# MAGIC %md A partir daí, podemos preparar os dados da seguinte forma:

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
from pyspark.sql.functions import min, max

# COMMAND ----------

# DBTITLE 1,Create Database
# MAGIC %sql
# MAGIC
# MAGIC DROP DATABASE IF EXISTS rodrigo_catalog.journey CASCADE;
# MAGIC CREATE DATABASE rodrigo_catalog.journey;
# MAGIC USE rodrigo_catalog.journey;

# COMMAND ----------

# DBTITLE 1,Initialize silver table paths
# MAGIC %sh
# MAGIC rm -r /dbfs/tmp/completejourney/silver/ 
# MAGIC mkdir -p /dbfs/tmp/completejourney/silver/

# COMMAND ----------

# DBTITLE 1,Transactions
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS rodrigo_catalog.journey.transactions')

# expected structure of the file
transactions_schema = StructType([
  StructField('household_id', IntegerType()),
  StructField('basket_id', LongType()),
  StructField('day', IntegerType()),
  StructField('product_id', IntegerType()),
  StructField('quantity', IntegerType()),
  StructField('sales_amount', FloatType()),
  StructField('store_id', IntegerType()),
  StructField('discount_amount', FloatType()),
  StructField('transaction_time', IntegerType()),
  StructField('week_no', IntegerType()),
  StructField('coupon_discount', FloatType()),
  StructField('coupon_discount_match', FloatType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/transaction_data.csv',
      header=True,
      schema=transactions_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .saveAsTable('rodrigo_catalog.journey.transactions')
  )

  # show data
display(
  spark.table('rodrigo_catalog.journey.transactions')
  )

# COMMAND ----------

# DBTITLE 1,Products
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS products')

# expected structure of the file
products_schema = StructType([
  StructField('product_id', IntegerType()),
  StructField('manufacturer', StringType()),
  StructField('department', StringType()),
  StructField('brand', StringType()),
  StructField('commodity', StringType()),
  StructField('subcommodity', StringType()),
  StructField('size', StringType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/product.csv',
      header=True,
      schema=products_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .saveAsTable('rodrigo_catalog.journey.products')
  )

# show data
display(
  spark.table('rodrigo_catalog.journey.products')
  )

# COMMAND ----------

# DBTITLE 1,Households
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS rodrigo_catalog.journey.households')

# expected structure of the file
households_schema = StructType([
  StructField('age_bracket', StringType()),
  StructField('marital_status', StringType()),
  StructField('income_bracket', StringType()),
  StructField('homeownership', StringType()),
  StructField('composition', StringType()),
  StructField('size_category', StringType()),
  StructField('child_category', StringType()),
  StructField('household_id', IntegerType())
  ])

# read data to dataframe
households = (
  spark
    .read
    .csv(
      '/tmp/completejourney/bronze/hh_demographic.csv',
      header=True,
      schema=households_schema
      )
  )

# make queryable for later work
households.createOrReplaceTempView('households')

# income bracket sort order
income_bracket_lookup = (
  spark.createDataFrame(
    [(0,'Under 15K'),
     (15,'15-24K'),
     (25,'25-34K'),
     (35,'35-49K'),
     (50,'50-74K'),
     (75,'75-99K'),
     (100,'100-124K'),
     (125,'125-149K'),
     (150,'150-174K'),
     (175,'175-199K'),
     (200,'200-249K'),
     (250,'250K+') ],
    schema=StructType([
            StructField('income_bracket_numeric',IntegerType()),
            StructField('income_bracket', StringType())
            ])
    )
  )

# make queryable for later work
income_bracket_lookup.createOrReplaceTempView('income_bracket_lookup')

# household composition sort order
composition_lookup = (
  spark.createDataFrame(
    [ (0,'Single Female'),
      (1,'Single Male'),
      (2,'1 Adult Kids'),
      (3,'2 Adults Kids'),
      (4,'2 Adults No Kids'),
      (5,'Unknown') ],
    schema=StructType([
            StructField('sort_order',IntegerType()),
            StructField('composition', StringType())
            ])
    )
  )

# make queryable for later work
composition_lookup.createOrReplaceTempView('composition_lookup')

# persist data with sort order data and a priori segments
(
  spark
    .sql('''
      SELECT
        a.household_id,
        a.age_bracket,
        a.marital_status,
        a.income_bracket,
        COALESCE(b.income_bracket_numeric, -1) as income_bracket_alt,
        a.homeownership,
        a.composition,
        COALESCE(c.sort_order, -1) as composition_sort_order,
        a.size_category,
        a.child_category
      FROM households a
      LEFT OUTER JOIN income_bracket_lookup b
        ON a.income_bracket=b.income_bracket
      LEFT OUTER JOIN composition_lookup c
        ON a.composition=c.composition
      ''')
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .saveAsTable('rodrigo_catalog.journey.households')
  )

# show data
display(
  spark.table('rodrigo_catalog.journey.households')
  )

# COMMAND ----------

# DBTITLE 1,Coupons
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS rodrigo_catalog.journey.coupons')

# expected structure of the file
coupons_schema = StructType([
  StructField('coupon_upc', StringType()),
  StructField('product_id', IntegerType()),
  StructField('campaign_id', IntegerType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/coupon.csv',
      header=True,
      schema=coupons_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .saveAsTable('rodrigo_catalog.journey.coupons')
  )

# show data
display(
  spark.table('rodrigo_catalog.journey.coupons')
  )

# COMMAND ----------

# DBTITLE 1,Campaigns
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS rodrigo_catalog.journey.campaigns')

# expected structure of the file
campaigns_schema = StructType([
  StructField('description', StringType()),
  StructField('campaign_id', IntegerType()),
  StructField('start_day', IntegerType()),
  StructField('end_day', IntegerType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/campaign_desc.csv',
      header=True,
      schema=campaigns_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .saveAsTable('rodrigo_catalog.journey.campaigns')
  )

# show data
display(
  spark.table('rodrigo_catalog.journey.campaigns')
  )

# COMMAND ----------

# DBTITLE 1,Coupon Redemptions
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS rodrigo_catalog.journey.coupon_redemptions')

# expected structure of the file
coupon_redemptions_schema = StructType([
  StructField('household_id', IntegerType()),
  StructField('day', IntegerType()),
  StructField('coupon_upc', StringType()),
  StructField('campaign_id', IntegerType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/coupon_redempt.csv',
      header=True,
      schema=coupon_redemptions_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .saveAsTable('rodrigo_catalog.journey.coupon_redemptions')
  )

# show data
display(
  spark.table('rodrigo_catalog.journey.coupon_redemptions')
  )

# COMMAND ----------

# DBTITLE 1,Campaign-Household Relationships
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS rodrigo_catalog.journey.campaigns_households')

# expected structure of the file
campaigns_households_schema = StructType([
  StructField('description', StringType()),
  StructField('household_id', IntegerType()),
  StructField('campaign_id', IntegerType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/campaign_table.csv',
      header=True,
      schema=campaigns_households_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .saveAsTable('rodrigo_catalog.journey.campaigns_households')
  )

# show data
display(
  spark.table('rodrigo_catalog.journey.campaigns_households')
  )

# COMMAND ----------

# DBTITLE 1,Causal Data
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS rodrigo_catalog.journey.causal_data')

# expected structure of the file
causal_data_schema = StructType([
  StructField('product_id', IntegerType()),
  StructField('store_id', IntegerType()),
  StructField('week_no', IntegerType()),
  StructField('display', StringType()),
  StructField('mailer', StringType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/causal_data.csv',
      header=True,
      schema=causal_data_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .saveAsTable('rodrigo_catalog.journey.causal_data')
  )

# show data
display(
  spark.table('rodrigo_catalog.journey.causal_data')
  )

# COMMAND ----------

# MAGIC %md ## Passo 2: Ajustar Dados Transacionais
# MAGIC
# MAGIC Com os dados brutos carregados, precisamos fazer alguns ajustes nos dados transacionais. Embora este conjunto de dados esteja focado em campanhas gerenciadas pelo varejista, a inclusão de informações de correspondência de desconto de cupom indicaria que os dados de transação refletem descontos originados tanto de cupons gerados pelo varejista quanto pelo fabricante. Sem a capacidade de vincular um produto-transação específico a um cupom específico (quando ocorre um resgate), assumiremos que qualquer valor de *coupon_discount* associado a um valor de *coupon_discount_match* diferente de zero origina-se de um cupom do fabricante. Todos os outros descontos de cupom serão assumidos como sendo de cupons gerados pelo varejista.
# MAGIC
# MAGIC Além da separação dos descontos de cupom do varejista e do fabricante, calcularemos um valor de lista para um produto como o valor de vendas menos todos os descontos aplicados:

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC USE CATALOG rodrigo_catalog;
# MAGIC USE DATABASE journey;

# COMMAND ----------

# DBTITLE 1,Adjusted Transactions
# MAGIC %sql
# MAGIC
# MAGIC DROP TABLE IF EXISTS transactions_adj;
# MAGIC
# MAGIC CREATE TABLE transactions_adj
# MAGIC USING DELTA
# MAGIC AS
# MAGIC   SELECT
# MAGIC     household_id,
# MAGIC     basket_id,
# MAGIC     week_no,
# MAGIC     day,
# MAGIC     transaction_time,
# MAGIC     store_id,
# MAGIC     product_id,
# MAGIC     amount_list,
# MAGIC     campaign_coupon_discount,
# MAGIC     manuf_coupon_discount,
# MAGIC     manuf_coupon_match_discount,
# MAGIC     total_coupon_discount,
# MAGIC     instore_discount,
# MAGIC     amount_paid,
# MAGIC     units
# MAGIC   FROM (
# MAGIC     SELECT 
# MAGIC       household_id,
# MAGIC       basket_id,
# MAGIC       week_no,
# MAGIC       day,
# MAGIC       transaction_time,
# MAGIC       store_id,
# MAGIC       product_id,
# MAGIC       COALESCE(sales_amount - discount_amount - coupon_discount - coupon_discount_match,0.0) as amount_list,
# MAGIC       CASE 
# MAGIC         WHEN COALESCE(coupon_discount_match,0.0) = 0.0 THEN -1 * COALESCE(coupon_discount,0.0) 
# MAGIC         ELSE 0.0 
# MAGIC         END as campaign_coupon_discount,
# MAGIC       CASE 
# MAGIC         WHEN COALESCE(coupon_discount_match,0.0) != 0.0 THEN -1 * COALESCE(coupon_discount,0.0) 
# MAGIC         ELSE 0.0 
# MAGIC         END as manuf_coupon_discount,
# MAGIC       -1 * COALESCE(coupon_discount_match,0.0) as manuf_coupon_match_discount,
# MAGIC       -1 * COALESCE(coupon_discount - coupon_discount_match,0.0) as total_coupon_discount,
# MAGIC       COALESCE(-1 * discount_amount,0.0) as instore_discount,
# MAGIC       COALESCE(sales_amount,0.0) as amount_paid,
# MAGIC       quantity as units
# MAGIC     FROM transactions
# MAGIC     );
# MAGIC     
# MAGIC SELECT * FROM transactions_adj;

# COMMAND ----------

# MAGIC %md ## Passo 3: Explorar os Dados
# MAGIC
# MAGIC As datas exatas de início e fim dos registros neste conjunto de dados são desconhecidas. Em vez disso, os dias são representados por valores entre 1 e 711, o que parece indicar os dias desde o início do conjunto de dados:

# COMMAND ----------

# DBTITLE 1,Household Data in Transactions
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   COUNT(DISTINCT household_id) as uniq_households_in_transactions,
# MAGIC   MIN(day) as first_day,
# MAGIC   MAX(day) as last_day
# MAGIC FROM transactions_adj;

# COMMAND ----------

# MAGIC %md Um foco principal da nossa análise será como os domicílios respondem a várias campanhas de varejistas, que podemos assumir incluir mala direta direcionada e cupons. Nem todos os domicílios no conjunto de dados de transações foram alvo de uma campanha, mas todos os domicílios que foram alvo estão representados no conjunto de dados de transações:

# COMMAND ----------

# DBTITLE 1,Household Data in Campaigns
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   COUNT(DISTINCT a.household_id) as uniq_households_in_transactions,
# MAGIC   COUNT(DISTINCT b.household_id) as uniq_households_in_campaigns,
# MAGIC   COUNT(CASE WHEN a.household_id==b.household_id THEN 1 ELSE NULL END) as uniq_households_in_both
# MAGIC FROM (SELECT DISTINCT household_id FROM transactions_adj) a
# MAGIC FULL OUTER JOIN (SELECT DISTINCT household_id FROM campaigns_households) b
# MAGIC   ON a.household_id=b.household_id

# COMMAND ----------

# MAGIC %md Quando cupons são enviados para um domicílio como parte de uma campanha, os dados indicam quais produtos estão associados a esses cupons. A tabela *coupon_redemptions* nos fornece detalhes sobre quais desses cupons foram resgatados em quais dias por um determinado domicílio. No entanto, o próprio cupom não é identificado em um determinado item de linha de transação.
# MAGIC
# MAGIC Em vez de trabalhar na associação de itens de linha específicos de volta aos resgates de cupons e, assim, vincular transações a campanhas específicas, optamos por simplesmente atribuir todos os itens de linha associados a produtos promovidos por campanhas como afetados pela campanha, independentemente de um resgate de cupom ter ocorrido. Isso é um pouco impreciso, mas estamos fazendo isso para simplificar nossa lógica geral. Em uma análise do mundo real desses dados, **essa é uma simplificação que deve ser revista**. Além disso, observe que não estamos examinando a influência de displays na loja e folhetos específicos da loja (capturados na tabela *causal_data*). Novamente, estamos fazendo isso para simplificar nossa análise.
# MAGIC
# MAGIC A lógica mostrada aqui ilustra como associaremos campanhas a compras de produtos e será reproduzida em nosso caderno de engenharia de recursos:

# COMMAND ----------

# DBTITLE 1,Transaction Line Items Flagged for Promotional Influences
# MAGIC %sql
# MAGIC
# MAGIC WITH 
# MAGIC     targeted_products_by_household AS (
# MAGIC       SELECT DISTINCT
# MAGIC         b.household_id,
# MAGIC         c.product_id
# MAGIC       FROM campaigns a
# MAGIC       INNER JOIN campaigns_households b
# MAGIC         ON a.campaign_id=b.campaign_id
# MAGIC       INNER JOIN coupons c
# MAGIC         ON a.campaign_id=c.campaign_id
# MAGIC       )
# MAGIC SELECT
# MAGIC   a.household_id,
# MAGIC   a.day,
# MAGIC   a.basket_id,
# MAGIC   a.product_id,
# MAGIC   CASE WHEN a.campaign_coupon_discount > 0 THEN 1 ELSE 0 END as campaign_coupon_redemption,
# MAGIC   CASE WHEN a.manuf_coupon_discount > 0 THEN 1 ELSE 0 END as manuf_coupon_redemption,
# MAGIC   CASE WHEN a.instore_discount > 0 THEN 1 ELSE 0 END as instore_discount_applied,
# MAGIC   CASE WHEN b.brand = 'Private' THEN 1 ELSE 0 END as private_label,
# MAGIC   CASE WHEN c.product_id IS NULL THEN 0 ELSE 1 END as campaign_targeted
# MAGIC FROM transactions_adj a
# MAGIC INNER JOIN products b
# MAGIC   ON a.product_id=b.product_id
# MAGIC LEFT OUTER JOIN targeted_products_by_household c
# MAGIC   ON a.household_id=c.household_id AND 
# MAGIC      a.product_id=c.product_id

# COMMAND ----------

# MAGIC %md Uma última coisa a ser observada, este conjunto de dados inclui dados demográficos apenas para cerca de 800 dos 2.500 domicílios encontrados no histórico de transações. Esses dados serão úteis para fins de perfil, mas precisamos ter cuidado antes de tirar conclusões de uma amostra tão pequena dos dados.
# MAGIC
# MAGIC Da mesma forma, não temos detalhes sobre como os 2.500 domicílios no conjunto de dados foram selecionados. Todas as conclusões tiradas de nossa análise devem ser vistas levando em consideração essa limitação:

# COMMAND ----------

# DBTITLE 1,Households with Demographic Data
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   COUNT(DISTINCT a.household_id) as uniq_households_in_transactions,
# MAGIC   COUNT(DISTINCT b.household_id) as uniq_households_in_campaigns,
# MAGIC   COUNT(DISTINCT c.household_id) as uniq_households_in_households,
# MAGIC   COUNT(CASE WHEN a.household_id==c.household_id THEN 1 ELSE NULL END) as uniq_households_in_transactions_households,
# MAGIC   COUNT(CASE WHEN b.household_id==c.household_id THEN 1 ELSE NULL END) as uniq_households_in_campaigns_households,
# MAGIC   COUNT(CASE WHEN a.household_id==c.household_id AND b.household_id==c.household_id THEN 1 ELSE NULL END) as uniq_households_in_all
# MAGIC FROM (SELECT DISTINCT household_id FROM transactions_adj) a
# MAGIC LEFT OUTER JOIN (SELECT DISTINCT household_id FROM campaigns_households) b
# MAGIC   ON a.household_id=b.household_id
# MAGIC LEFT OUTER JOIN households c
# MAGIC   ON a.household_id=c.household_id
