# Master-Project-
Mesure et amélioration de la vitesse d’inférence d’un réseau neuronal convolutif sur plateformes matérielles embarquées à faibles coûts

Par Samuel Kong Tung Lin

Résumé:
Ce projet consiste à mesurer et améliorer la vitesse d’inférence d’un réseau de neurones de type CNN dans la classification d’images sur des systèmes embarquées à faibles coûts soit le Jetson Nano de Nvidia, le Coral Dev Board de Google et le RaspberryPi 4 de Raspberry Pi. Il y aura la description de chaque plateforme et les restrictions de leur utilisation. Ces plateformes sont nouvellement disponibles et pourront prendre une grande place au milieu des environnements de l’internet des objets. 

Dans ce projet, il y aura la revue de littérature de méthodes de réduction de complexité et l’implémentation de plusieurs réseaux sur les plateformes embarquées. Les méthodes incluent principalement la perforation et la séparation de la couche convolutives et l’élimination de poids du model. La phase d’expérimentation est séparée en deux, une pour l’entrainement des systèmes avec la librairie TensorFlow et l’autre pour l’inférence avec la librairie TensorFlow Lite qui est optimisé pour les systèmes ayants un pouvoir de calcul faible. Les expériences seront faites avec la base de données CIFAR10, qui inclus 60000 images en couleurs de 10 classes différentes.  

Ensuite, il y aura la comparaison des résultats des réseaux entre les plateformes pour pouvoir analyser les différences entre eux et mesurer l’impact de la réduction de complexité selon la plateforme embarquée. Les mesures de performances seront le temps d’inférence, le temps d’entrainement, la taille et la précision du model. Des recommandations sur l’utilisations des plateformes matérielles embarquées sera faite selon leurs applications et l’analyse des résultats. 
