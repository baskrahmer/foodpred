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

- `/app`: code that enters the Docker image (Dockerfile is at root for wider buidl context) (`onnx`, AWS Lambda)
- `/harrygobert`: source code for finetuning Transformer model (`transformers`, `torch`, `lightning`)
- `/terraform`: infrastructure code (`terraform`)
- `/frontend`: minimal web frontend (`react`)
- `/.github`: CI/CD jobs using OIDC

# Evaluation and limitations

Creating a unified benchmark for this task is a work in progress. Right now the selection metric is just a simple top-1
validation accuracy for 5-fold cross-validation over the entire dataset, using the fixed seed in the code. Whichever
hyperparameter configuration receives the highest score is then subsequently
trained on the whole dataset and packaged into production format.

There are several limitations to this approach, most importantly the schewed language distribution since more than half
of the dataset products are French, and for this reason the current model does not perform as good for other languages.

# Prod model

The production model is a fine-tuned version
of [distilbert-base-multilingual-cased](https://huggingface.co/distilbert-base-multilingual-cased).
The training curve can be seen
via [this WANDB report](https://wandb.ai/baskra/harrygobert/reports/Foodpred-prod-training--Vmlldzo0NDM3MTUx). The
hyperparameter configuration used achieves a top-1 accuracy of 0.77 using 5-fold cross-validation as described above.

# EcoScore calculation

The displayed score is calculated based on the implementation of
OpenFoodFacts ([code snippet](https://github.com/openfoodfacts/openfoodfacts-server/blob/main/lib/ProductOpener/Ecoscore.pm#L1019-L1049)):

```perl
# Formula to transform the Environmental Footprint single score to a 0 to 100 scale
# Note: EF score are for mPt / kg in Agribalyse, we need it in micro points per 100g

# Milk is considered to be a beverage
if (has_tag($product_ref, 'categories', 'en:beverages')
    or (has_tag($product_ref, 'categories', 'en:milks')))
{
    # Beverages case: score = -36*\ln(x+1)+150score=− 36 * ln(x+1) + 150
    $product_ref->{ecoscore_data}{agribalyse}{is_beverage} = 1;
    $product_ref->{ecoscore_data}{agribalyse}{score}
        = round(-36 * log($agribalyse{$agb}{ef_total} * (1000 / 10) + 1) + 150);
}
else {
    # 2021-02-17: new updated formula: 100-(20 * ln(10*x+1))/ln(2+ 1/(100*x*x*x*x))  - with x in MPt / kg.
    $product_ref->{ecoscore_data}{agribalyse}{is_beverage} = 0;
    $product_ref->{ecoscore_data}{agribalyse}{score} = round(
        100 - 20 * log(10 * $agribalyse{$agb}{ef_total} + 1) / log(
            2 + 1 / (
                      100 * $agribalyse{$agb}{ef_total}
                    * $agribalyse{$agb}{ef_total}
                    * $agribalyse{$agb}{ef_total}
                    * $agribalyse{$agb}{ef_total}
            )
        )
    );
}
if ($product_ref->{ecoscore_data}{agribalyse}{score} < 0) {
    $product_ref->{ecoscore_data}{agribalyse}{score} = 0;
}
elsif ($product_ref->{ecoscore_data}{agribalyse}{score} > 100) {
    $product_ref->{ecoscore_data}{agribalyse}{score} = 100;
```

Basically, for most products the Eco-Score formula is:

`100 - (20 * ln(10EF + 1)) / ln(2 + 1/(100 * EF^4))`

Whereas for beverages it is:

`- 36 * ln(EF+1) + 150`

Additional product-specific attributes are added as positive or negative bonus points to arrive to a final normalised
score. This score is thresholded to fall within a range of 0 to 100. On [foodpred.com](https://foodpred.com/), the bonus
points and thresholding steps are currently not classified and thus omitted. This means the raw score is displayed
without bonus/minus points and without this normalisation. For reference, after bonus/minus points, a score between 0
and 20 would have a label of E and a score between 80 and 100 would have a label of A.
