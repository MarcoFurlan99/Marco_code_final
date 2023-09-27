class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.new_best = True

    def early_stop(self, validation_loss):
        """when
        validation_loss > self.min_validation_loss + self.min_delta
        for enough consecutive times (patience)
        then returns True"""
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.new_best = True
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.new_best = False
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
