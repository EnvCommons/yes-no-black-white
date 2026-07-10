from openreward.environments import Server

from env import YesNoBlackWhiteEnvironment

if __name__ == "__main__":
    server = Server([YesNoBlackWhiteEnvironment])
    server.run()
