[![codecov](https://codecov.io/gh/baskrahmer/harrygobert/branch/master/graph/badge.svg?token=HW8LS4VSEP)](https://codecov.io/gh/baskrahmer/harrygobert)

# Introduction

This is a static dump of my side-project to predict the LCI category of foods based on their name, and subsequently use
this data give an indication of the Eco-Score of that product. It is a work in progress and the current live version can
be viewed at [foodpred.com](https://www.foodpred.com/) (Lambda has a warmup of ~10s). Although not my most polished nor
finished work; it is a good demonstration of my full-stack skills and the most recent project I can freely share.

# Repo contents

There are a couple of important directories to this work:

- `/.github`: CD pipelines
- `/app`: code that enters the Docker image (Dockerfile is at root for wider buidl context)
- `/harrygobert`: source code for finetuning Transformer model
- `/terraform`: infrastructure code
- `/frontend`: minimal React code

# Next steps

As said, this is a work in progress. To give some insights on what I would improve (in order of descending importance):

- **The predictive backend**: the dataset is quite unbalanced across languages and categories, which needs to be
  optimised.
  I have before used an untrained model with a cosine distance-based approach to the category names; this works good but
  does not utilise the data for training. Experiments need to follow.
- **The frontend**: it is very minimal. There is more data to be displayed for each category such as what contributes
  most to this specific CO2 score and a conversion to the actual Eco-Score.
- **Testing and reliability**: speaks for itself; is too lean now because project is still in the PoC phase.
