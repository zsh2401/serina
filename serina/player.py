import threading

import pygame


class Player:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load("./声音玩具_你的城市.mp3")
        pygame.mixer.music.set_volume(0.8)
        pygame.mixer.music.play()
        music_thread = threading.Thread(target=self.tick, args=('your_music_file.mp3',))
        music_thread.start()
        self.pause_music()

    def pause_music(self):
        pygame.mixer.music.pause()

    def unpause_music(self):
        pygame.mixer.music.unpause()

    def tick(self):
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
