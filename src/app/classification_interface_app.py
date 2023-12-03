import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import pandas as pd

class ClassificationViewer(tk.Frame):
    def __init__(self, parent, classifications):
        super().__init__(parent)
        self.pack(fill="both", expand=True)
        self.create_widgets(classifications)

    def create_widgets(self, classifications):
        self.canvas = tk.Canvas(self)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.table_frames = []

        # Caminhar por cada classificação
        for title, table, accuracy, model_name in classifications:
            frame = ttk.Frame(self.scrollable_frame)
            frame.pack(side="top", fill="x", padx=10, pady=5)

            # Mostrando nome do model
            model_label = tk.Label(frame, text=f'Model: {model_name}')
            model_label.pack(pady=5)

            # Mostrando a tabela
            table_tree = ttk.Treeview(frame, columns=list(table.columns), show='headings')
            for col in table.columns:
                table_tree.heading(col, text=col)
                table_tree.column(col, width=100, anchor='center')

            for _, row in table.iterrows():
                table_tree.insert('', 'end', values=row.tolist())

            table_tree.pack(expand=True, fill='both')

            # Display accuracy
            accuracy_label = tk.Label(frame, text=f'Accuracy: {accuracy:.2%}')
            accuracy_label.pack(pady=5)

            self.table_frames.append(frame)

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.update_idletasks()

        self.canvas.configure(scrollregion=self.canvas.bbox("all"), yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

def display_classification_window(classifications):
    root = tk.Tk()
    app = ClassificationViewer(root, classifications)
    root.mainloop()