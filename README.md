# LLM school project for AI RAG | Projet de RAG pour LLM

"ENG"

In this project, you serve the llama3.2 model a text extracted from a PDF file from a dropbox cloud to create context and get an answer with specifics informations coming from the served file. The model does not give the same response depending if you ask the same question with or without context. 

The retrieved file is language oriented and you can create prompt to see the difference between the answers.

It is possible to change the creativity of the model through the temperature prop inside the query_model_with_rag function inside the ollama_serving.py file. It also affect the answer given by the model.

---

"FR"

Dans ce projet, vous servez au modèle llama3.2 un texte extrait d'un fichier PDF stocké sur un cloud Dropbox afin de créer un contexte et d'obtenir une réponse contenant des informations spécifiques issues du fichier fourni. Le modèle ne donne pas la même réponse selon que vous posez la même question avec ou sans contexte.

Le fichier récupéré est orienté sur les langues, et vous pouvez créer des prompts pour observer la différence entre les réponses.

Il est possible de modifier la créativité du modèle grâce à la propriété temperature dans la fonction query_model_with_rag située dans le fichier ollama_serving.py. Cela influence également la réponse donnée par le modèle.

---

To setup the project, you need to install the libraries inside the requirements with the following command | Pour mettre en place le projet, il vous faut installer les librairies suivantes :

``` shell
pip install -r requirements.txt
```

Have Ollama installed on your Desktop/Laptop and the specific llama3.2 model setup | Avoir Ollama et le model llama3.2 sur votre support (PC/PC Portable).

Here is the link to download Ollama | Voici le lien pour télécharge Ollama : https://ollama.com/download

Once, the basic setup installed, you need to execute those commands | Après avoir configuré le projet, il vous faut exécuter les commandes suivantes :

``` shell
python script.py # To download the PDF file from dropbox and extract the text

python ollama_serving.py # To serve the text to the model and send a prompt 
```

If the file is not donwloaded you can still make a request | Dans le cas où le fichier n'est pas télécharger, il est toujours possible d'executer un prompt.
