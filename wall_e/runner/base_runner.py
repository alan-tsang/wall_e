from ..callback.base_callback import BaseCallBack


class RunnerBase:

    def __init__(self):
        self.callbacks = []

    def fit(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def valid(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass

    def register_callbacks(self, callbacks):
        self.callbacks = []
        for callback in callbacks:
            if callback is None:
                continue
            assert isinstance(callback, BaseCallBack), \
                "Callbacks must be subclass of BaseCallBack."
            callback.on_register()
            self.callbacks.append(callback)

    def unregister_callbacks(self):
        self.callbacks = []
        return True

    def on_exception(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_exception(*args, **kwargs)

    def before_fit(self):
        for callback in self.callbacks:
            callback.before_fit()

    def after_fit(self):
        for callback in self.callbacks:
            callback.after_fit()

    def before_train(self):
        for callback in self.callbacks:
            callback.before_train()

    def after_train(self):
        for callback in self.callbacks:
            callback.after_train()

    def before_running_batch(self):
        for callback in self.callbacks:
            callback.before_running_batch()

    def after_running_batch(self):
        for callback in self.callbacks:
            callback.after_running_batch()

    def before_running_epoch(self):
        for callback in self.callbacks:
            callback.before_running_epoch()

    def after_running_epoch(self):
        for callback in self.callbacks:
            callback.after_running_epoch()

    def before_valid(self):
        for callback in self.callbacks:
            callback.before_valid()

    def after_valid(self):
        for callback in self.callbacks:
            callback.after_valid()

    def before_test(self):
        for callback in self.callbacks:
            callback.before_test()

    def after_test(self):
        for callback in self.callbacks:
            callback.after_test()
