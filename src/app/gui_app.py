import cv2
import tkinter as tk

from tkinter import filedialog, ttk
from PIL import Image, ImageTk

class Application:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Inicializar variáveis para controle de zoom
        self.zoom_factor = 1.0
        self.zoom_levels = [0.25, 0.5, 1.0, 2.0, 4.0]

        # Definir o tamanho inicial da janela e centralizá-la na tela
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        initial_width = int(screen_width * 0.8)
        initial_height = int(screen_height * 0.8)
        x = (screen_width - initial_width) // 2
        y = (screen_height - initial_height) // 2
        self.window.geometry(f"{initial_width}x{initial_height}+{x}+{y}")

        # Criar um menu
        self.menu = tk.Menu(self.window)
        self.window.config(menu=self.menu)

        # Adicionar opções ao menu
        self.file_menu = tk.Menu(self.menu)
        self.menu.add_cascade(label="Arquivo", menu=self.file_menu)
        self.file_menu.add_command(label="Abrir", command=self.load_image)

        # Adicionar uma entrada para o valor de N
        self.n_label = tk.Label(self.window, text="Valor de N:")
        self.n_label.pack()
        self.n_entry = tk.Entry(self.window)
        self.n_entry.pack()

        self.options_menu = tk.Menu(self.menu)
        self.menu.add_cascade(label="Opções", menu=self.options_menu)
        self.options_menu.add_command(label="Segmentar núcleos", command=self.segmentation)
        self.options_menu.add_command(label="Caracterizar núcleos", command=self.characterize)
        self.options_menu.add_command(label="Classificar núcleos", command=self.classification)

        # Criar um frame para conter a imagem e o texto
        self.image_frame = tk.Frame(self.window)
        self.image_frame.pack(expand=True, fill="both")

        # Configurar o frame para centralizar a imagem e o texto
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)

        # Adicionar uma barra de rolagem
        self.scrollbar_h = ttk.Scrollbar(self.image_frame, orient="horizontal")
        self.scrollbar_h.grid(row=1, column=0, sticky="ew")
        self.scrollbar_v = ttk.Scrollbar(self.image_frame, orient="vertical")
        self.scrollbar_v.grid(row=0, column=1, sticky="ns")

        # Adicionar um canvas para exibir a imagem
        self.canvas = tk.Canvas(self.image_frame, bd=0, xscrollcommand=self.scrollbar_h.set, yscrollcommand=self.scrollbar_v.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Configurar a barra de rolagem para controlar o canvas
        self.scrollbar_h.config(command=self.canvas.xview)
        self.scrollbar_v.config(command=self.canvas.yview)

        # Inicializar a imagem como None
        self.cv_img = None
        self.photo = None

        # Inicializar variáveis de texto
        self.error_n_label = tk.Label(self.image_frame, text="")
        self.error_n_label.grid(row=2, column=0, pady=10)

        self.text_label = tk.Label(self.image_frame, text="")
        self.text_label.grid(row=3, column=0, pady=10)

        self.filename_label = tk.Label(self.image_frame, text="")
        self.filename_label.grid(row=4, column=0)

        # Iniciar o loop da janela TKinter
        self.window.mainloop()

    def segmentation(self):
        """
        Abrir um campo para digitar N, para segmentar a imagem no tamanho desejado.
        Usuário pode usar o padrão = 100.
        """
        try:
            self.error_n_label.config(text="")
            n_value = int(self.n_entry.get())
        except (ValueError, UnboundLocalError):
            self.error_n_label.config(text="Você não inseriu N, valor a ser considerado = 100")
            n_value = 100
        finally:
            # TODO: Aplicar para a imagem que vai ser inserida, ou seja: recortar a imagem que vai ser inserida (Buscar no CSV)
            # TODO: INSERIR IMAGEM -> RECORTAR CELULAS DA IMAGEM -> SEGMENTAR AS CELULAS -> MOSTRAR PARA O USUARIO (PLOTAR OU COLOCAR NA INTERFACE)
            pass

    def characterize(self):
        print('Caracterizar o núcleo através de descritores de forma.')
    
    def classification(self):
        print('Classificar cada núcleo encontrado na imagem.')

    def load_image(self):
        '''
        Configuração da entrada da imagem.
            - Aplicar funcionalidade de zoom
            - Mostrar o nome do arquivo
        '''
        # Abrir o seletor de arquivos
        file_path = filedialog.askopenfilename()

        if file_path:
            self.cv_img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

            filename = file_path.split("/")[-1]

            # Atualizar os textos das seleções do arquivo
            self.text_label.config(text="IMAGEM SELECIONADA:")
            self.filename_label.config(text=f"Nome do Arquivo: {filename}")

            # Criar uma imagem TKinter a partir da imagem OpenCV
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.cv_img))

            self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, anchor="center", image=self.photo)

            self.canvas.config(scrollregion=self.canvas.bbox("all"))

            # Adicionar eventos de zoom
            self.canvas.bind("<Enter>", self.enable_zoom)
            self.canvas.bind("<Leave>", self.disable_zoom)
            self.canvas.bind("<MouseWheel>", self.zoom)

    def enable_zoom(self, event):
        self.canvas.bind("<MouseWheel>", self.zoom)

    def disable_zoom(self, event):
        self.canvas.unbind("<MouseWheel>")

    def zoom(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def zoom_in(self):
        if self.zoom_factor < self.zoom_levels[-1]:
            self.zoom_factor *= 2
            self.update_zoom()

    def zoom_out(self):
        if self.zoom_factor > self.zoom_levels[0]:
            self.zoom_factor /= 2
            self.update_zoom()

    def update_zoom(self):
        # Redimensionar a imagem
        new_width = int(self.cv_img.shape[1] * self.zoom_factor)
        new_height = int(self.cv_img.shape[0] * self.zoom_factor)
        resized_img = cv2.resize(self.cv_img, (new_width, new_height))

        # Criar uma nova imagem TKinter e atualizar o canvas
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized_img))
        self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, anchor="center", image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))