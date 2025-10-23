import tkinter as tk
from tkinter import messagebox
from main import train_tree, text_to_binary_features, predict_single_with_heuristic

class SpamGUI:
    def __init__(self):
        self.tree = train_tree()
        self.root = tk.Tk()
        self.root.title("ID3 Spam Filter")
        self.root.geometry("550x350")
        self.root.resizable(False, False)
        self._build_ui()

    def _build_ui(self):
        tk.Label(
            self.root,
            text="Provera poruke",
            font=("Arial", 14, "bold")
        ).pack(pady=10)

        self.text_entry = tk.Text(self.root, height=6, width=60, wrap=tk.WORD, font=("Arial", 11))
        self.text_entry.pack(pady=10)

        tk.Button(
            self.root,
            text="Proveri poruku",
            command=self._classify_message,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            relief="raised",
            padx=10,
            pady=5
        ).pack()

        self.result_label = tk.Label(self.root, text="", font=("Arial", 14, "bold"))
        self.result_label.pack(pady=20)

    def _classify_message(self):
        msg = self.text_entry.get("1.0", tk.END).strip()
        if not msg:
            messagebox.showwarning("Upozorenje", "Unesite tekst poruke!")
            return
        feats = text_to_binary_features(msg)
        feats = {k: str(v) for k, v in feats.items()}
        label = predict_single_with_heuristic(self.tree, feats)
        if label == "spam":
            self.result_label.config(text="Rezultat: SPAM 🚫", fg="red")
        else:
            self.result_label.config(text="Rezultat: HAM ✅", fg="green")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = SpamGUI()
    gui.run()
