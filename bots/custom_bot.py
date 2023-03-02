from screen import Screen



class BotBase:

    def __init__(self):
        self.screen = Screen()
        
        self.state = None


    def set_state(self, image):
        """
        Extracts state from image.
        """
        raise NotImplementedError
    
    def set_actions(self, state):
        """
        Choose actions for given state.
        """
        raise NotImplementedError
    
    def play_actions(self, actions):
        """
        Execute given actions.
        """
        raise NotImplementedError
    
    @staticmethod
    def detect_game_screen(image):
        """
        Detect what screen (in game, home, etc..) is visible.
        """
        raise NotImplementedError
    
    @staticmethod
    def is_game_end(image):
        """
        Returns True when on victory/loss screen.
        """
        raise NotImplementedError
    
    @staticmethod
    def is_victory(image):
        """
        Returns True if image is victory screen.
        """
        raise NotImplementedError
    
    def run(self, auto_play):
        if auto_play:
            # navigate from current screen to in game
            raise NotImplementedError
        
        image = self.screen.take_screenshot()
        while not self.is_game_end(image):

            self.set_state(image)
            self.set_actions(self.state)
            self.play_actions(self.actions)

            image = self.screen.take_screenshot()

        victory = self.is_victory(image)
