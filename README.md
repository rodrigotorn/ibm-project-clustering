## Python-Clustering

### Definição do problema

Esse projeto foi apresentado como trabalho final para a certificação em ciência de dados ofertada pela IBM, por isso os arquivos estão em inglês.

A premissa consiste em: dado uma pessoa que precisa se mudar de cidade mas gosta muito do seu bairro atual, como escolher um bairro em outra cidade que tenha características parecidas com a do seu bairro atual?

O estudo de caso foi realizado especificamente para as cidades de Nova York - USA e Toronto - CAN. As características levadas em consideração foram a quantidade e os tipos de estabelecimento que estão contidos dentro de cada bairro. Para obter tais informações foi utilizada a API do Foursquare, que permite obter os estabelecimentos em uma localização e raio de distância definidos.

O clustering foi realizado utilizando o algoritmo K-Means, dividindo os bairros em 5 clusters diferentes. A plotagem em um mapa interativo foi feita utilizando Folium, para conseguir ver os mapas é necessário sinalizar o notebook como confiável.

### Arquivos

* clustering.py: Jupyter Notebook em pypercent format contendo todos os códigos.
* clustering.html : Relatório do projeto. Contem as informações resumidas e sem códigos.

### Referências

[1] Coursera - IBM Data Science Certificate.
