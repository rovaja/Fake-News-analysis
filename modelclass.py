import torch
import torchmetrics
from torch.optim import lr_scheduler, AdamW
from torch import nn
import lightning as L
from transformers import DistilBertForSequenceClassification


class NewsClassifier(L.LightningModule):
    """News article classifier based on BERT pretrained models."""
    backbone = {
        "distilbert": (DistilBertForSequenceClassification, "distilbert-base-uncased"),
        }
    archs = {
        "distilbert_reg": ('distilbert', 'reg_clf_head'),
        "distilbert": ('distilbert', 'default_clf_head'),
    }
    def __init__(
                self, 
                train_data: torch.utils.data.Dataset,
                val_data: torch.utils.data.Dataset,
                test_data: torch.utils.data.Dataset,
                lr: float = 1e-4,
                arch: str = "distilbert",
                batch_size: int = 32,
                backbone_tuning: bool = False):
        super(NewsClassifier, self).__init__()
        self.save_hyperparameters()
        self.arch, self.clf_arch = self.archs[arch]
        model, version = self.backbone[self.arch]
        
        self.lr=lr
        self.batch_size = batch_size
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.train_score = torchmetrics.classification.AUROC(task='binary')
        self.valid_score =torchmetrics.classification.AUROC(task='binary')
        self.test_score = torchmetrics.classification.AUROC(task='binary')
        self.model = model.from_pretrained(
            version,
            return_dict=True,
            output_attentions=True,
            num_labels=1,
            )
        self._clf_head_structure()
        self.backbone_tuning = backbone_tuning
        self._base_grad(requires_grad_clf = True, requires_grad=self.backbone_tuning)
        self.loss_function = nn.BCEWithLogitsLoss()
        
    def _clf_head_structure(self):
        """Set up classifier head structure"""
        if self.clf_arch == 'reg_clf_head':
            in_features = self.model.classifier.in_features
            out_features = self.model.classifier.out_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, in_features),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(in_features, out_features),
            )
            
    
    def _base_grad(
        self,
        requires_grad_clf: bool,
        requires_grad: bool = False
        ):
        """Freeze or unfreeze model for gradient computing."""
        for param in self.model.__getattr__(self.arch).parameters():
            param.requires_grad = requires_grad
        for param in self.model.classifier.parameters():
            param.requires_grad = requires_grad_clf

    def _common_step(self, batch):
        """Shared step between training, validation and testing."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["target"]
        y_hat, _ , logit = self(input_ids, attention_mask)
        logit = logit.squeeze(-1)
        y_hat = y_hat.squeeze(-1)
        loss = self.loss_function(logit, labels)
        return loss, y_hat, labels.int()

    def forward(self, input_ids, attention_mask):
        """Prediction method of the model."""
        out = self.model(input_ids, attention_mask=attention_mask)
        probabilities = torch.sigmoid(out.logits)
        
        return probabilities, out.attentions, out.logits

    def training_step(self, batch, batch_idx):
        """Training step"""
        loss, y_hat, labels = self._common_step(batch)
        
        self.train_score.update(y_hat, labels)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log_dict({"train_score": self.train_score}, on_epoch=True, on_step=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        loss, y_hat, labels = self._common_step(batch)
        self.valid_score.update(y_hat, labels)
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        self.log_dict({"val_score": self.valid_score}, on_epoch=True, on_step=False)
        
        return loss

    def test_step(self, batch, batch_idx):
        """Test step"""
        loss, y_hat, labels = self._common_step(batch)
        self.test_score.update(y_hat, labels)
        self.log_dict({'test_loss': loss,"test_score": self.test_score}, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Select optimizer algorithm for the model."""
        optimizer = AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)
        lr_scheduler = self.fetch_scheduler(optimizer, scheduler_type='CosineAnnealingLR')
        return {"optimizer" : optimizer, 'lr_scheduler' : lr_scheduler}
    
    def fetch_scheduler(self, optimizer, scheduler_type = None):
        """Set scheduler for the optimizer learning rate."""

        if scheduler_type == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max = 2 * len(self.train_data)/self.batch_size,
                eta_min=self.lr/100
            )

        elif scheduler_type == 'CosineAnnealingWarmRestarts':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=2*self.batch_size,
                eta_min=self.lr/100
            )

        elif scheduler_type == None:
            return None

        return scheduler

    def train_dataloader(self):
        """Train dataset generator"""
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True)

    def val_dataloader(self):
        """Validation dataset generator"""
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True)

    def test_dataloader(self):
        """Test dataset generator"""
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True)