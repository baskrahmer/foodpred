[![codecov](https://codecov.io/gh/baskrahmer/harrygobert/branch/master/graph/badge.svg?token=HW8LS4VSEP)](https://codecov.io/gh/baskrahmer/harrygobert)

Source code of [foodpred.com](https://foodpred.com/), a web application to compute the ecological impact of consumer
goods in the food industry. The predictions are classifications of the [CIQUAL](https://ciqual.anses.fr/) dataset based
on products from [openfoodfacts](https://openfoodfacts.org/) in all languages. The predictions only look at the text of
the product. That means factors such as transport are not tailored to your location or the specific brand, but rather
taken as an average over all available products.

# Use case

For a lot of products in many languages, there is reliable and good-quailtiy data available. The purpose of foodpred is
not to replace this data, but to provide an educated guess when such data is not available. Examples include slight
variations in product names, typos, and different languages. The use case is then also more in the realm of automated
matching, where manual classification is not a feasible option. When you are looking at just a handful of products, I
recommend doing the classification yourself.

This project is in development and the public API is subject to change. The project is self-funded.

# Repository contents

- `/app`: code that enters the Docker image (Dockerfile is at root for wider buidl context) (`onnx`, `AWS Lambda`)
- `/harrygobert`: source code for finetuning Transformer model (`transformers`, `torch`, `lightning`)
- `/terraform`: infrastructure code (`terraform`)
- `/frontend`: minimal web frontend (`react`)

# Benchmarks

Creating a unified benchmark for this task is a work in progress. 
