dev endpoint: https://api.81181825.xyz/?product=test

prod endpoint: https://api.foodpred.com/?product=test

Example Agribalyse snippet:

```yaml
1000:
  DQR:
    GR: '4.41'
    P: '2.49'
    TeR: '3.54'
    TiR: '2.73'
    overall: '3.29'
  LCI_name: Pastis (anise-flavoured spirit)
  Livraison: Ambiant (moyenne)
  Preparation: "R\xE9frig\xE9r\xE9 chez le consommateur"
  avion: false
  ciqual_AGB: '1000'
  ciqual_code: '1000'
  groupe: boissons
  impact_environnemental:
    Acidification terrestre et eaux douces:
      etapes:
        Agriculture: 0
        Consommation: 3.4793328e-05
        Emballage: 0.0046828017
        "Supermarch\xE9 et distribution": 6.354403e-05
        Transformation: 0.0019775257
        Transport: 0.00051901142
      synthese: 0.0072777644
      unite: mol H+ eq/kg de produit
    Appauvrissement de la couche d'ozone:
      etapes:
        Agriculture: 0
        Consommation: 0.0041383883
        Emballage: 0.072623762
        "Supermarch\xE9 et distribution": 0.0072456904
        Transformation: 0.058563635
        Transport: 0.040536261
      synthese: 0.18307514
      unite: E-06 kg CVC11 eq/kg de produit
    Changement climatique:
      etapes:
        Agriculture: 0
        Consommation: 0.0047993021
        Emballage: 0.49690685
        "Supermarch\xE9 et distribution": 0.012364537
        Transformation: 0.4658128
        Transport: 0.17690961
      synthese: 1.1565882
      unite: kg CO2 eq/kg de produit
    Eutrophisation eaux douces:
      etapes:
        Agriculture: 0
        Consommation: 0.0031349554
        Emballage: 0.095903725
        "Supermarch\xE9 et distribution": 0.0055021285999999996
        Transformation: 0.049883325
        Transport: 0.014808111
      synthese: 0.16924558
      unite: E-03 kg P eq/kg de produit
    Eutrophisation marine:
      etapes:
        Agriculture: 0
        Consommation: 0.0077991547
        Emballage: 0.6752250599999999
        "Supermarch\xE9 et distribution": 0.014130594
        Transformation: 0.41523097999999997
        Transport: 0.10196678999999999
      synthese: 1.2144347
      unite: E-03 kg N eq/kg de produit
    Eutrophisation terreste:
      etapes:
        Agriculture: 0
        Consommation: 6.8499657e-05
        Emballage: 0.0078500547
        "Supermarch\xE9 et distribution": 0.00012579701
        Transformation: 0.0056624946
        Transport: 0.0011344856
      synthese: 0.014841716
      unite: mol N eq/kg de produit
    Formation photochimique d'ozone:
      etapes:
        Agriculture: 0
        Consommation: 0.014117292
        Emballage: 2.0425077000000003
        "Supermarch\xE9 et distribution": 0.028058467
        Transformation: 5.8903899
        Transport: 0.42364452
      synthese: 8.3989108
      unite: E-03 kg NMVOC eq/kg de produit
    Particules:
      etapes:
        Agriculture: 0
        Consommation: 0.0003222193
        Emballage: 0.061298028000000004
        "Supermarch\xE9 et distribution": 0.00056291189
        Transformation: 0.068069619
        Transport: 0.011337907000000001
      synthese: 0.14163342
      unite: E-06 disease inc./kg de produit
    Rayonnements ionisants:
      etapes:
        Agriculture: 0
        Consommation: 0.047839148
        Emballage: 0.053489378
        "Supermarch\xE9 et distribution": 0.078789182
        Transformation: 0.12508201
        Transport: 0.013281877
      synthese: 0.31848138
      unite: kBq U-235 eq/kg de produit
    Score unique EF:
      etapes:
        Agriculture: 0
        Consommation: 0.0024293397
        Emballage: 0.051649676000000005
        "Supermarch\xE9 et distribution": 0.0044194823
        Transformation: 0.050783404000000004
        Transport: 0.01375452
      synthese: 0.12303632999999999
      unite: mPt/kg de produit
    Utilisation du sol:
      etapes:
        Agriculture: 0
        Consommation: 0.023721083
        Emballage: 5.50476
        "Supermarch\xE9 et distribution": 0.04211235
        Transformation: 13.898953
        Transport: 1.8447158
      synthese: 21.322845
      unite: Pt/kg de produit
    "\xC9cotoxicit\xE9 pour \xE9cosyst\xE8mes aquatiques d'eau douce":
      etapes:
        Agriculture: 0
        Consommation: 0.25081812
        Emballage: 8.5288162
        "Supermarch\xE9 et distribution": 0.44024315
        Transformation: 6.275886
        Transport: 1.9167451
      synthese: 17.415179
      unite: CTUe/kg de produit
    "\xC9puisement des ressources eau":
      etapes:
        Agriculture: 0
        Consommation: 0.0097571262
        Emballage: 0.12327471
        "Supermarch\xE9 et distribution": 0.039989272
        Transformation: 0.078515905
        Transport: 0.018608459
      synthese: 0.27016888
      unite: m3 depriv./kg de produit
    "\xC9puisement des ressources min\xE9raux":
      etapes:
        Agriculture: 0
        Consommation: 0.042713564
        Emballage: 1.2145978
        "Supermarch\xE9 et distribution": 0.07225273
        Transformation: 0.33740524
        Transport: 0.5335022700000001
      synthese: 2.2005789
      unite: E-06 kg Sb eq/kg de produit
    "\xC9puisement des ressources \xE9nerg\xE9tiques":
      etapes:
        Agriculture: 0
        Consommation: 1.0090332
        Emballage: 7.2666296
        "Supermarch\xE9 et distribution": 1.7229013
        Transformation: 7.2959482
        Transport: 2.7018105
      synthese: 19.992829
      unite: MJ/kg de produit
  materiau_emballage: Verre
  nom_francais: Pastis
  saison: mix de consommation FR
  sous_groupe: "boisson alcoolis\xE9es"
```
