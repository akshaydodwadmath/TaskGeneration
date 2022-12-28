from abc import ABC, abstractmethod
from itertools import chain


class AbstractAgent(ABC):
    def __init__(self, encoder, decoder, vocab=None, device='cpu'):
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab
        self.device = device

    @abstractmethod
    def get_parameters(self):
        return chain(self.encoder.parameters(), self.decoder.parameters())

    @abstractmethod
    def set_train(self):
        self.encoder.train()
        self.decoder.train()

    @abstractmethod
    def set_eval(self):
        self.encoder.eval()
        self.decoder.eval()

    @abstractmethod
    def load_model(self, *args):
        pass

    @abstractmethod
    def save_model(self, *args):
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        """
        You need a concrete method that does the forward pass for your specific architecture
        """
        raise NotImplementedError

    @abstractmethod
    def do_batch(self, type):
        """
        You need a method to return a batch of test_data
        """
        raise NotImplementedError

    @abstractmethod
    def get_evaluation_data(self, *args, **kwargs):
        raise NotImplementedError
