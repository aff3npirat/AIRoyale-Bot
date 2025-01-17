from device.controller import Controller
from device.state.screens import ScreenDetector
from constants import CARD_HEIGHT, CARD_WIDTH, CARD_CONFIG



class BotBase:

    def __init__(self, hash_size=8, device=5555):
        self.controller = Controller(device=device)
        self.screen = ScreenDetector(hash_size)

        self.replay_buffer = []
    
    @staticmethod
    def slot_to_xy(slot):
        x, y = CARD_CONFIG[slot+1][:2]
        x = x + CARD_WIDTH/2
        y = y + CARD_HEIGHT/2

        return x, y
    
    def init_model(self, path):
        raise NotImplementedError
    
    @staticmethod
    def get_illegal_actions(state):
        """
        Returns indices of illegal actions.
        """
        raise NotImplementedError
    
    @staticmethod
    def exp_to_dict(experience):
        """
        Returns an episode entry returned from `with_reward` as dictionary.
        """
        if experience is None:
            board, context, action, reward, done = None
        else:
            (board, context), action, reward, done = experience
        return {"board": board, "context": context, "action": action, "reward": reward, "done": done}
    
    @staticmethod
    def get_reward(*args, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def with_reward(episode, victory):
        """
        Returns triplets of <state, action, reward>
        """
        raise NotImplementedError
    
    def store_experience(self, state, actions):
        """
        Stores state, actions pair in replay buffer.
        """
        raise NotImplementedError

    def get_state(self, image):
        """
        Extracts state from image.
        """
        raise NotImplementedError
    
    def get_actions(self, state, eps):
        """
        Choose actions for given state.
        """
        raise NotImplementedError
    
    def play_actions(self, actions):
        """
        Execute given actions.
        """
        raise NotImplementedError
    
    def run(self, *args, **kwargs):
        raise NotImplementedError
