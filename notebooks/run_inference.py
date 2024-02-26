# Databricks notebook source
import mlflow

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(f"runs:/f1dcc98c0cfe4ea8a2009dd96b3d704c/checkpoint-300")

# COMMAND ----------

questions = [
    " The coffee shop 'The Wrestlers' is located on the riverside, near 'Raja Indian Cuisine'. They serve English food and a price range of less than £20, and are not family-friendly. ",
    " Cotto is an inexpensive English restaurant near The Portland Arms in the city centre, and provides English coffee shop food. Customers recently rated the store 5 out of 5. ",
    " The Eagle coffee shops Chinese food, moderately priced, customer rating 1 out of 5, located city centre, kid friendly, located near Burger King. ",
    " The Punter is a child friendly establishment located by the riverside with a customer rating of 1 out of 5. ",
    " Taste of Cambridge, a coffee shop specializing in English eatery, is located in riverside near Crowne Plaza Hotel and is known to be very kid friendly. ",
    " The Punter is an expensive Chinese coffee shop located near Café Sicilia. ",
    " Clowns is a coffee shop that severs English food. Clowns is located in Riverside near Clare Hall. Clowns customer service ratings are low. ",
]

# COMMAND ----------

import mlflow

# Make a prediction using the loaded model
loaded_model.predict(
    {"prompt": questions,
     "max_tokens": [512]*len(questions),
    }
)


# Make a prediction using the loaded model
# loaded_model.predict(
#     {"prompt": "The Restaurant Gymkhana near Marlybenone station has a high customer star rating and offers a unique Indian cuisines",
#      "temperature": 0.4,
#      "max_tokens": 128,
#     }
# )

# COMMAND ----------


