import tkinter as tk
import tkinter.simpledialog as simpledialog

from client import Client


class Interface:

    def _handle_send(self):
        text = self.entry_text.get()
        if not text:
            return

        response = self.client.send_prediction(text)
        self.label_result.config(fg='green' if int(response['label']) else 'red')
        self.label_result_text.set(
            f"{'Positive' if int(response['label']) else 'Negative'} at {round(float(response['score']), 2)}")

    def _handle_exit(self):
        self.root.destroy()

    def __init__(self, root: tk.Tk):
        self.root = root

        self.label = tk.Label(text="Sentiment prediction")
        self.label.pack()

        self.entry_text = tk.Entry(width=50)
        self.entry_text.pack()

        self.label_result_text = tk.StringVar()
        self.label_result = tk.Label(textvariable=self.label_result_text)
        self.label_result.pack()

        self.frame_buttons = tk.Frame()
        self.button_exit = tk.Button(
            text="Exit", master=self.frame_buttons, command=self._handle_exit)
        self.button_exit.pack(side=tk.RIGHT)
        self.button_send = tk.Button(
            text="Send", master=self.frame_buttons, command=self._handle_send)
        self.button_send.pack(side=tk.RIGHT)
        self.frame_buttons.pack()

        root.withdraw()
        self.answer = simpledialog.askstring(
            "Url", "Enter connection url", initialvalue="http://127.0.0.1:5000/")
        self.client = Client()
        self.client.connect(self.answer)
        root.wm_deiconify()


if __name__ == "__main__":
    window = tk.Tk()
    window.eval('tk::PlaceWindow . center')
    Interface(window)
    window.mainloop()
