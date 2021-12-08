# MLOPS : Modèle de machine learning mis en production

Mise en production un modèle de machine learning d'analyse de sentiments

Le déploiement du modèle devra se faire automatique via un script simple. 

Il faudra que le modèle puisse gérer une certaine charge. Vous pourrez soit faire du batch processing / une API ou du streaming. 

Il faudra que le système soit capable de : 

- soit de se réentraîner tout seul régulièrement grâce à des données nouvellement labelisée
- soit de faire remonter des alertes si il existe des risques que le modèle ne marche plus (distributional shift)

Bonus : le modèle sera packagé dans un conteneur docker
