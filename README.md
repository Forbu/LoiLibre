# LoiLibre
LoiLibre : interface open source pour de l'assistance juridique

Bienvenue à LoiLibre, un assistant juridique open source conçu pour rendre l'accès aux informations juridiques plus accessible à tous. 
Avec LoiLibre, nous espérons éliminer la complexité souvent intimidante des lois françaises, en offrant une plateforme facile à utiliser pour naviguer et comprendre les lois et règlements.
Que vous soyez un avocat, un étudiant en droit, ou simplement un citoyen souhaitant comprendre vos droits, LoiLibre est conçu pour vous. 
Ce README.md est destiné à vous aider à démarrer avec LoiLibre, en vous fournissant les informations clés sur le projet et les instructions pour sa configuration et son utilisation.
Nous sommes impatients de voir comment LoiLibre aidera à démocratiser l'accès aux informations juridiques pour tous.


Ce projet utilise d'autres projets open source (open science juridique : les données des différents codes viennent de droit.fr ou encore de legifrance.gouv.fr), nous les remercions.

### Le site internet 

Il s'agit d'un site internet rudimentaire (je ne suis pas webdev ...) qui permettra d'interagir avec l'assitant juridique.

Si vous avez des propositions d'amélioration du site vous pouvez ajouter des issues etc 


### Le model

Ici nous utilisons les évolutions récentes des modèles de langage (ex : chatGPT) pour créer un assitant capable d'intéragir avec des données (ici le corpus juridique).

Les détails du code seront fait plus tard.

Le modèle de l'assistant juridique est simplement chatGPT (via l'API en ligne) couplé à une interface qui permet à chatGPT d'avoir des informations concernant des articles juridiques en lien avec la question posée.

Il s'agit ici d'un modèle simple, dans le futur nous avons plusieurs possibilité autour du fine tuning de modèles open source ou encore d'autres approches. 



