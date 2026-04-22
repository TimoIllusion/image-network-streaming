from abc import ABC, abstractmethod


class BackendInterface(ABC):

    @abstractmethod
    def send_frame_to_ai_server(self, frame):
        pass
    
    @abstractmethod
    def close(self):
        pass
