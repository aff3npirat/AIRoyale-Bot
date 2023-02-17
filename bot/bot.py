from state.game_states import PLAYING, OTHER



class Bot:
    
    def __init__(self):
        self.state = None
        self.game_state = OTHER

    def detect_game_state(self):
        raise NotImplementedError

    def choose_action(self):
        raise NotImplementedError

    def play_action(self, action):
        raise NotImplementedError

    def set_state(self):
        """
        Extracts state from game.
        """
        raise NotImplementedError

    def run(self):
        while self.detect_game_state() == PLAYING:
            self.set_state()

            action = self.choose_action()

            self.play_action(action)