# Projeto para análise e monitoramento de posturas

---

 ## Preparando o ambiente

- Até o momento (12/23), a biblioteca MediaPipe (versão 0.10.2) funciona com o Python na versão 3.8 ~ 3.11.
- Crie uma pasta onde vão ser armazenadas todas as imagens a ser analisadas.
- Faça o download do modelo desejado do MediaPipe Pose.  Disponível em: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models
- Faça o download das bibliotecas utilizadas com: `pip install -r requirements.txt`

Ajustando programa

No desenvolvimento do programa foi utilizado o [python-dotenv](https://pypi.org/project/python-dotenv/), então, o ajuste no programa pode ser feito adicionando um arquivo `.env` com as mesmas variáveis, ou colocar o caminho dos arquivos dentro de cada arquivo `.py`.

Para o `.env`:

```
MODEL_PATH_FULL=\caminho\para\modelo\. . .\modelo_escolhido.task
MODEL_PATH_HEAVY=\caminho\para\modelo\. . .\modelo_escolhido.task
IMAGE_DATASET_FOLDER_1=\caminho\para\pasta\com\imagens\. . .\imagens_para_analise
MODEL_IMAGE=\caminho\para\imagem\modelo\. . .\imagens_para_analise\imagem.jpeg
```

Caso opte por não usar o `.env`, vá aos arquivos `gerar_dataset.py` e `main.py` e atualize o conteúdo das variáveis associadas ao dotenv.

O `MODEL_PATH_FULL` está sendo utilizado para análise em tempo real, e o `MODEL_PATH_HEAVY` está sendo utilizado para análise de imagens, porém, pode-se utilizar o mesmo modelo para ambas as análises.
A `IMAGE_DATASET_FOLDER_1` é a pasta com as imagens a serem analisadas.
O `MODEL_IMAGE` é uma imagem (podendo ser uma das imagens analisadas) que será disponibilizada para o usuário de basear ao permanecer na postura.


No arquivo `main.py` (na linha 539) deve ser ajustado o `cv2.VideoCapture(0)` para usar a captura da câmera desejada, sendo que 0 (zero) define a captura para webcam do computador.

Usando o programa

Depois de ajustar o programa, a primeira coisa que deve ser feita é uma coleta de várias imagens da mesma postura para fazer a análise. Lembrando que, apenas com as alterações e ajustes citados acima, o programa realiza a análise e monitoramento apenas do torso (braços, ombro e pescoço). Todas essas imagens devem estar dentro da pasta de imagens para serem analisadas.
Após a coleta de imagens, basta rodar o arquivo `gerar_dataset.py` para extrair dados das imagens. Ao final vai ser gerado um arquivo `.csv` com os dados de todas as imagens, esse arquivo `.csv` vai ser utilizado pelo arquivo `main.py`.
Feito as coletas e os ajustes do programa, basta rodar o arquivo `main.py`, este criar duas janelas, uma com a imagem modelo para referência, e outra com o monitoramento da postura em tempo real. Para encerrar o programa é necessário selecionar a janela com a análise em tempo real e pressionar `q`, ou, apertar `ctrl` + `c` no prompt.
