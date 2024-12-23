"""
Global config shared across all the processes to keep track of if the application is running or not.
Used in background listeners to keep checking if the application is running or shutting down.
"""

from multiprocessing import Manager

manager = Manager()
shared_config = manager.dict()
shared_config["running"] = False
