## Documentação dos códigos

Para fazer a primeira parte do projeto:

1. Os dados usados no treinamento e teste dos classificadores e do segmentador devem ser preprocessados. Através da planilha classifications.csv, obtenha as coordenadas dos núcleos das células das imagens disponíveis no dataset (apenas uma parte das imagens está disponível). Recorte as imagens, gerando uma sub-imagem 100x100 para cada núcleo e armazene em sub-diretórios de acordo com a sua classe. O nome da imagem deve ser o número da célula na planilha.
  - Arquivo [app/image_processing.py](https://github.com/brunofaria27/image-processing/blob/main/src/app/image_processing.py) 
  - ***Criar as pastas:*** na função `create_folders_classes` para armazenar as imagens segmentadas em **100x100** ou como o usuário desejar em **NxN** (se passado por parametro).
  - ***Leitura do arquivo:*** na função `processing_images` faz-se a leitura do arquivo como um Dicionário para que as colunas sejam as chaves e chama-se a função `processing_image_scale` onde o código irá iterar sobre cada linha do CSV.
  - ***Processamento da imagem:*** irá pegar da tabela qual a coordenada do núcleo e fazer um cálculo básico para recortar a partir do centro do núcleo **NxN** pixels. Porém tem que tratar a questão do tamanho do recorte para os 4 lados ***(NORTE, SUL, LESTE E OESTE)***, pois pode acabar ficando fora dos limites da imagem fazendo com que ele não fique na dimensão correta.
  - ***Tratamento da dimensão:*** bastou fazer `if's` para testar se o limite foi superado para algum lado, se caso fosse você teria que puxar o recorte para o outro lado com a quantidade de pixels faltantes, assim o recorte ficaria sempre correto.

2. Implemente um ambiente totalmente gráfico com um menu para as seguintes funcionalidades:
 - Ler e visualizar imagens nos formatos PNG e JPG. As imagens podem ter qualquer resolução.
 - Segmentar os núcleos das células contidas nas imagens e recortar uma região NxN ao redor do centro do núcleo. A princípio N=100, mas pode ser alterado.
 - Caracterizar o núcleo através de descritores de forma.
 - Classificar cada núcleo encontrado na imagem.

3. Implemente a funcionalidade de leitura e exibição das imagem com opção de zoom.
  - Arquivo [app/gui.py](https://github.com/brunofaria27/image-processing/blob/main/src/app/gui.py)
  - Para a interface foi usado `tkinter` onde criou-se uma classe com todas importantes funções que a interface deve ter, além disso tem-se as funções que devem ser chamadas ao clicar nos itens do `Menu` para fazer todas as funcionalidades acima [2].

4. Implemente a funcionalidade de segmentação dos núcleos. Compare o resultado medindo a distância entre o centro do núcleo segmentado e o que está na planilha.
------------------------
## CRIC Cervix Cell Classification - [CSV Description](https://github.com/brunofaria27/image-processing/blob/main/src/data/classifications.csv)

400 images from microscope slides of the uterine cervix using the conventional smear (Pap smear) and the epithelial cell abnormalities classified according to Bethesda system.

### Data Fields
- `image_id`
  This is the integer that identifies the image at http://database.cric.com.br/.
- `image_filename`
  This is the name that identifies the image in the ZIP file that you have.
- `image_doi`
  This is the DOI that identifies the image.
- `cell_id`
  This is the integer that identifies the cell at http://database.cric.com.br/.
- `bethesda_system`
  Classification of the cell
  using the Bethesda system.
  It is on of the following:
  - Negative for intraepithelial lesion
  - ASC-US
    Atypical squamous cells of undetermined significance
  - ASC-H
    Atypical squamous cells cannot exclude HSIL
  - LSIL
    Low grade squamous intraepithelial lesion
  - HSIL
    High grade squamous intraepithelial lesion
  - SCC
    Squamous cell carcinoma
- `nucleus_x`
  Integer between 1 and 1384 equal to coordinate x of the pixel that represent the cell.
- `nucleus_y`
  Integer between 1 and 1384 equal to coordinate y of the pixel that represent the cell.