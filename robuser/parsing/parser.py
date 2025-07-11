from abc import abstractmethod, ABC


class Parser(ABC):
    def __init__(self, data_path):
        self.data_path = data_path

    @abstractmethod
    def run_parser(self):
        """

        Returns:
            A dictionary with the following format:
            {
                '/path/to/audio_file_1.wav': {'emotion': 'class_name', ...},
                ...
            }
        """
        pass
