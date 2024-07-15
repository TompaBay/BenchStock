class Exp(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        print(self.device)
        

    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _build_train_setting(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        device = self.args.gpu
        return device

    def _get_data(self):
        pass

    def process_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def train_epoch(self):
        pass

    def test(self):
        pass
