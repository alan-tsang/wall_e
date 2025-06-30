class BaseCallBack:
    """
    CallBack的基类，基于训练的生命周期设计。
    """

    def __init__(self, runner):
        self.runner = runner

    def on_register(self):
        """
        Called when the callback is registered.
        """
        self.runner.logger.info(f"Callback {self.__class__.__name__} registered.")

    def on_exception(self, *args, **kwargs):
        """
        Called when an exception is raised.
        """
        pass

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_running_epoch(self):
        """
        Called before each epoch.
        """
        pass

    def after_running_epoch(self):
        """
        Called after each epoch.
        """
        pass

    def before_running_batch(self):
        """
        Called before each iteration.
        """
        pass

    def after_running_batch(self):
        """
        Called after each iteration.
        """
        pass

    def before_valid(self):
        pass

    def after_valid(self):
        pass

    def before_test(self):
        pass

    def after_test(self):
        pass

    def before_fit(self):
        """
        Called before the training starts.
        """
        pass

    def after_fit(self):
        """
        Called after the training ends.
        """
        pass
