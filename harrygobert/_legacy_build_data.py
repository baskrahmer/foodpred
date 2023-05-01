"""
This script parses all OpenFoodFacts products that have a corresponding Agribalyse product. In essence, this is similar
to scraping the following URLs:
- https://nl.openfoodfacts.org/categories-properties/agribalyse-food-code-known
- https://nl.openfoodfacts.org/categories-properties/agribalyse-proxy-food-code-known

These products can be used to train a model that makes this mapping on its own. To get the Robotoff products, run the
following script (instructions in script docstring):
- https://github.com/openfoodfacts/robotoff/blob/master/scripts/category_dataset.py
"""
import os
from collections import defaultdict
from pathlib import Path

import requests
import yaml


def main(out_path=Path(__file__).parent / '../data'):
    # Load Robotoff products
    robotoff_products_path = 'https://static.openfoodfacts.net/data/taxonomies/categories.full.json'
    robotoff_products = requests.get(robotoff_products_path).json()  # 11434 products

    # Load Jumbo categories
    jumbo_to_off_path = Path(os.path.join(out_path, 'jumbo_to_off_mapping.yaml'))
    jumbo_categories = yaml.safe_load(open(jumbo_to_off_path))

    # Load OFF data
    off_url = 'https://raw.githubusercontent.com/openfoodfacts/openfoodfacts-server/main/taxonomies/categories.txt'
    off_taxonomies = requests.get(off_url).text

    # Load Agribalyse data
    agribalyse_url = 'https://raw.githubusercontent.com/datagir/ecolab-alimentation/master/data/out/Agribalyse.json'
    agribalyse_data = requests.get(agribalyse_url).json()
    ciqual_code_dict = {int(p['ciqual_code']): p for p in agribalyse_data}

    # Parse products
    products = parse_agribalyse(off_taxonomies)

    # Create parametrized mapping function
    off_to_agribalyse = lambda x: off_to_agribalyse_mapping(x, products=products)

    # Map Jumbo categories to Agribalyse
    map_jumbo_to_agribalyse(ciqual_code_dict, jumbo_categories, off_to_agribalyse)

    # Map Robotoff products to agribalyse
    product_to_agribalyse = map_robotoff_to_agribalyse(off_to_agribalyse, robotoff_products)

    # Save to disk
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    yaml.dump(product_to_agribalyse, open(os.path.join(out_path, 'product_to_ciqual.yaml'), 'w'))
    yaml.dump(ciqual_code_dict, open(os.path.join(out_path, 'ciqual_dict.yaml'), 'w'))

    # Create subdatasets and save to disk
    ciqual_to_ef_score = {k: v['impact_environnemental']['Score unique EF']['synthese'] for k, v in
                          ciqual_code_dict.items()}
    ciqual_to_ef_complete = {k: v['impact_environnemental'] for k, v in ciqual_code_dict.items()}
    ciqual_to_lci_name = {k: v['LCI_name'] for k, v in ciqual_code_dict.items()}
    yaml.dump(ciqual_to_ef_score, open(os.path.join(out_path, 'ciqual_to_ef_score.yaml'), 'w'))
    yaml.dump(ciqual_to_ef_complete, open(os.path.join(out_path, 'ciqual_to_ef_complete.yaml'), 'w'))
    yaml.dump(ciqual_to_lci_name, open(os.path.join(out_path, 'ciqual_to_lci_name.yaml'), 'w'))

    # Dict for textual category to EF
    tmp = {ciqual_to_lci_name[k]: v for k, v in ciqual_to_ef_score.items()}
    ef_dict = dict(sorted(tmp.items(), key=lambda x: x[1], reverse=True))

    # Dict for numeric category to EF
    tmp = {k: v for k, v in ciqual_to_ef_score.items()}
    ef_dict = dict(sorted(tmp.items(), key=lambda x: x[1], reverse=True))


def map_jumbo_to_agribalyse(ciqual_code_dict, categories, off_to_agribalyse):
    jumbo_to_agribalyse = dict()
    for cat in categories.values():
        if not cat:
            continue
        agribalyse_cat = off_to_agribalyse(cat['off_category'])
        if agribalyse_cat and agribalyse_cat in ciqual_code_dict:
            jumbo_to_agribalyse[cat['title'].split(' --> ')[-1]] = agribalyse_cat
    print(f'Total Jumbo mappings: {len(categories)}')
    print(
        f'Total mapped to Agribalyse: {len(jumbo_to_agribalyse)} ({round(100 * len(jumbo_to_agribalyse) / len(categories), 2)}%)')
    print(f'Total Agribalyse categories: {len(ciqual_code_dict)}')


def map_robotoff_to_agribalyse(off_to_agribalyse, train):
    product_to_agribalyse = {}

    # Keys are standardized form of 'name' value (lowercase, hyphens etc) and thus ignored
    for t in train.values():

        agribalyse_parents = [off_to_agribalyse(p) for p in t.get('parents', []) if off_to_agribalyse(p)]
        if not agribalyse_parents:
            continue

        for lang, product_name in t['name'].items():
            if lang not in product_to_agribalyse:
                product_to_agribalyse[lang] = {}
            product_to_agribalyse[lang][product_name] = agribalyse_parents

    return product_to_agribalyse


def parse_agribalyse(off_to_agribalyse_data):
    synonyms_dict = defaultdict(dict)
    stopwords_dict = defaultdict(dict)

    curr_product = set()
    canonical = None
    products = {}

    for line in off_to_agribalyse_data.split('\n'):

        # Ignore comments
        if line.startswith('#'):
            continue

        # Parse synonyms
        elif line.startswith('synonyms'):
            _, language, synonyms = line.split(':')
            synonyms = [s.strip() for s in synonyms.split(',')]
            for s in synonyms:
                synonyms_dict[language][s] = set(synonyms)

        # Parse stopwords
        elif line.startswith('stopwords'):
            _, language, stopwords = line.split(':')
            stopwords = [s.strip() for s in stopwords.split(',')]
            for s in stopwords:
                stopwords_dict[language][s] = set(stopwords)

        # Attributes of current product
        elif line:

            line = line.split(':')
            attribute, value = line[0], ':'.join(line[1:])

            for v in value.split(', '):

                v = v.lower().replace(' ', '-')
                if attribute.startswith('<'):
                    pass  # TODO any parent-specific processing goes here
                elif canonical is None:
                    canonical = ':'.join([attribute, v])

                curr_product.add(':'.join([attribute, v]))

        # End of current product
        else:

            if curr_product:
                products[canonical] = curr_product

            # End of product
            curr_product = set()
            canonical = None

    return products


def off_to_agribalyse_mapping(off_cat, products):
    """Return an empty string if no Agribalyse mapping exists"""

    if off_cat not in products:
        return ''
    product = products[off_cat]

    for attribute in product:
        if 'food_code' in attribute and 'agribalyse' in attribute:
            return int(attribute.split(':')[-1].split('_')[0])

    for attribute in product:
        if attribute.startswith('<'):
            parent_off_category = off_to_agribalyse_mapping(attribute[1:], products)
            if parent_off_category:
                return parent_off_category

    return ''


if __name__ == '__main__':
    main()
