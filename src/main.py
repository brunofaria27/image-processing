from app.gui import Application
from app.image_processing import main as image_processing_main

import tkinter as tk

if __name__ == "__main__":
    # Iniciar a interface gráfica
    app = Application(tk.Tk(), "Aplicativo de Processamento de Imagens")

    # Iniciar o processamento de imagem em segundo plano
    # image_processing_main()