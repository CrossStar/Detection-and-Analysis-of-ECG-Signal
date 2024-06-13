import json
import time

from PyQt5.QtCore import QThread, pyqtSignal, QMutex


class Worker(QThread):
    update_signal = pyqtSignal(str)

    def __init__(self, client, messages, model):
        super(Worker, self).__init__()
        self.client = client
        self.messages = messages
        self.model = model

    def run(self):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=True
        )
        content_accumulator = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content_accumulator += chunk.choices[0].delta.content
                self.update_signal.emit(chunk.choices[0].delta.content)



