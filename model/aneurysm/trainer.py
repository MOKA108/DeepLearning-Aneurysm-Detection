import torch

class AneurysmTrainer:
    """Training loop helper encapsulating one-epoch logic.

    Supports optional AMP (automatic mixed precision) when running on CUDA.
    """

    def __init__(self, model, optimizer, criterion, device, amp=False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.amp = amp and device == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    def train_one_epoch(self, loader):
        """Run one epoch of training over `loader`.

        Returns average loss over batches.
        """
        self.model.train()
        total_loss = 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            target_dtype = next(self.model.parameters()).dtype
            if X.dtype != target_dtype:
                X = X.to(target_dtype)
            self.optimizer.zero_grad(set_to_none=True)
            if self.amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(X).squeeze(1)
                    loss = self.criterion(logits, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(X).squeeze(1)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
