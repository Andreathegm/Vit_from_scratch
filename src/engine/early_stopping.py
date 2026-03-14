class EarlyStopping:

    def __init__(self, patience: int = 25):
        self.patience  = patience
        self.counter   = 0
        self.best_acc  = 0.0

    def step(self, val_acc: float) -> bool:
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter  = 0
        else:
            self.counter += 1
            self.status(self)

        return self.counter >= self.patience
    
    def status(self):
        print(f"EarlyStopping - Validation Accuracy not increasing since : {self.counter}/{self.patience}")

        