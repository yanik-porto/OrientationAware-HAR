import torch
from torch.utils.data import DataLoader
import os.path as op
from dvclive import Live
import time
from tqdm import tqdm

from encoder.dataset.tools.measure import *
from tools.evaluation import accuracy, accuracy_reconstruction, accuracy_2d_weighted, accuracy_multiple_labels, accuracy_text_embed, tad_accuracy
from tools.evaluation_det import getSingleStreamDetectionMAP
from tools.checkpoint import create_symlink
from language import Language
from encoder.dataset.dataloaders.formater import split_batch

class Trainer:
    def __init__(
            self,
            encoder: torch.nn.Module,
            head: torch.nn.Module,
            dl: DataLoader,
            optimizer: torch.optim,
            scheduler: torch.optim.lr_scheduler,
            config,
            out_train_path,
            device,
            epoch_start: int = 0,
            dl_val: DataLoader = None,
            gt_name: str = 'label',
            gpu_id_eval = 0,
            text_encoder = None,
            force_classif = False
    ):
        self.device = device
        self.encoder = encoder.to(self.device)
        self.head = head.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dl = dl
        self.config = config
        self.out_train_path = out_train_path
        self.force_classif = force_classif

        self.epoch_start = epoch_start
        self.dl_val = dl_val
        self.gt_name = gt_name
        self.gpu_id_eval = gpu_id_eval

        self.n_gpus = 1

        self.text_encoder = text_encoder
        self.language = Language(config, self.dl, self.dl_val, self.text_encoder) if 'language' in self.config else None

        self.print_loss_details = self.config["train"]["settings"].get("print_loss_details", False)

    def _encoder(self):
        return self.encoder

    def _head(self):
        return self.head

    def _run_batch(self, keypoints, labels, text_embeds=None):
        self.optimizer.zero_grad()

        feats = self.encoder(keypoints)
        if text_embeds is not None:
            feats = (feats, text_embeds)
        output = self.head(feats)

        if self.gt_name in ("binary_labels", "tad_labels", 'window_labels'):
            loss = self._head().loss_bce(output, labels)
        else:
            loss = self._head().loss(output, labels)

        loss.backward()
        self.optimizer.step()

        return float(loss.detach())


    def _run_epoch(self, ie):
        loss_epoch = AverageMeter()
        self.encoder.train()
        self.head.train()
        if self.text_encoder is not None:
            self.text_encoder.train()

        # update lambda with epoch if needed
        if hasattr(self._head(), 'update_lambda'):
            self._head().update_lambda(ie)

        # for batch in tqdm(self.dl):
        # pbar = tqdm(total=len(self.dl))#, position=0, leave=True)
        for ib, batch in enumerate(self.dl):
            keypoints, labels = split_batch(batch, self.device, self.gt_name)

            if self.language is not None:
                text_embed = self.language.get_feats_train(labels, batch)
                text_embed = text_embed.to(self.device)

            elif self.text_encoder is not None:
                if 'texts' in batch:
                    texts = batch['texts']
                    texts = texts[0]
                else:
                    texts = [self.dl.dataset.label_to_text(il, training=True) for il in labels]
                text_embed = self.text_encoder(texts)
            else:
                text_embed = None if not 'text_embed' in batch else batch['text_embed'].to(self.device)

            loss = self._run_batch(keypoints, labels, text_embed)
            loss_epoch.update(loss)

        #     pbar.update(ib)
        # pbar.close()

        print("Epoch #", ie, "(", len(self.dl), " iters | gpu #", self.device.index, ") | Loss : {loss:.5f}".format(loss=loss_epoch.avg), ' | lr : {lr:.5f}'.format(lr=self.scheduler.get_last_lr()[0]))
        self.scheduler.step()

        return loss_epoch.avg

    def _get_accuracy(self, keypoints, labels, text_embeds=None, meters=[]):
        feats = self.encoder(keypoints)
        if text_embeds is not None:
            feats = (feats, text_embeds)
        output = self.head(feats)
        if type(output) is tuple:
            output = output[0] 

        if self.gt_name == "tad_labels": # accuracy by class
            multi = True
            # gt = labels[0].cpu().numpy() if multi else torch.argmax(labels[0].data.cpu(), axis=2).numpy()
            bin_labels = labels[0].cpu() if type(labels) is tuple else labels.cpu()
            gt = torch.argmax(bin_labels, axis=2).numpy()
            # mask = output[1].to(bool)
            # preds = output[0][mask]
            # gt = gt[mask.cpu().numpy()]
            dmap_list, iou_list = getSingleStreamDetectionMAP(output, gt, multi=multi)
            meters["mAP@0.1"].update(dmap_list[0], 1) # only labels needed, no duration
            meters["mAP@0.25"].update(dmap_list[1], 1) # only labels needed, no duration
            meters["mAP@0.5"].update(dmap_list[2], 1) # only labels needed, no duration
        elif self.gt_name == "window_labels":
            probs = tad_accuracy(output, labels)
            meters["Top1"].update(probs.data.cpu().numpy()[0], bin_labels.numpy()[0]) # only labels needed, no duration

        else: # overall accuracy
            if text_embeds != None:
                if not self.force_classif:
                    [acc1, acc5], _ = accuracy_text_embed(output, labels, (1, 5))
                else:
                    assert len(output) == 4, str(len(output)) + " is different than 4"
                    acc1, acc5 = accuracy(output[3], labels, (1, 5))
                meters["Top1"].update(float(acc1.detach().cpu().numpy()), keypoints[0].size(0))
                meters["Top5"].update(float(acc5.detach().cpu().numpy()), keypoints[0].size(0))
            elif self.gt_name == "label":
                if len(output.shape) == 3:
                    M = output.shape[1]
                    output = output.flatten(0, 1)
                    if type(labels) is tuple:
                        labels = labels[0] 
                    labels = labels.repeat_interleave(M)
                acc1, acc5 = accuracy(output, labels, (1, 5))
                meters["Top1"].update(float(acc1.detach().cpu().numpy()), keypoints[0].size(0))
                meters["Top5"].update(float(acc5.detach().cpu().numpy()), keypoints[0].size(0))
            elif self.gt_name == "keypoint_gt":
                acc1 = accuracy_2d_weighted(output, labels)
                meters["Top1"].update(float(acc1.detach().cpu().numpy()), keypoints[0].size(0))
            elif self.gt_name == "binary_labels":
                acc1 = accuracy_multiple_labels(output.detach().cpu(), labels, topk=(1,))[0]
                meters["Top1"].update(float(acc1.detach().cpu().numpy()), keypoints[0].size(0))
            elif self.gt_name == "orientation":
                acc1 = accuracy_reconstruction(output, labels)
                meters["Top1"].update(float(acc1.detach().cpu().numpy()), keypoints[0].size(0))

    @torch.no_grad()
    def _run_eval(self):
        metrics = {}
        if self.gt_name == "window_labels":
            metrics["Top1"] = APMeter()
        elif self.gt_name == "tad_labels":
            metrics["mAP@0.1"] = AverageMeter()
            metrics["mAP@0.25"] = AverageMeter()
            metrics["mAP@0.5"] = AverageMeter()
        else:
            metrics["Top1"] = AverageMeter()
            metrics["Top5"] = AverageMeter()
            
        self.encoder.eval()
        self.head.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()

        for batch in self.dl_val:
            keypoints, labels = split_batch(batch, self.device, self.gt_name)
            
            text_embeds = None
                        
            if self.language is not None:
                text_embeds = self.language.get_feats_test(batch)
                text_embeds = text_embeds.to(self.device)

            elif self.text_encoder is not None:
                if 'texts' in batch:
                    texts = batch['texts']
                    texts = [text[0] for text in texts] # collate function convert list of string to list of list
                else:
                    texts = self.dl_val.dataset.label_map
                text_embeds = self.text_encoder(texts)
            elif 'text_embed' in batch:
                text_embeds = torch.from_numpy(self.dl_val.dataset.text_embeds).to(self.device)

            self._get_accuracy(keypoints, labels, text_embeds, metrics)

        print("Eval (", len(self.dl_val), " iters | gpu #", self.device.index, ")", end="")
        for metric_name, metric_value in metrics.items():
            print(f" | {metric_name}: {metric_value.value():.2f}", end="")
        print("")
        return next(iter(metrics.values())).value()

    def _save_checkpoint(self, ie, is_best=False):
        chkpt_path = op.join(self.out_train_path, "epoch_" + str(ie) + ".pth")
            
        tobesaved = {
                'epoch': ie,
                'lr': self.scheduler.get_last_lr(),
                'optimizer': self.optimizer.state_dict(),
                'encoder': self._encoder().state_dict(),
                'head': self._head().state_dict()
            }
        if self.text_encoder is not None:
            tobesaved['text_encoder'] = self.text_encoder.model.state_dict()
        torch.save(tobesaved, chkpt_path)

        # symlink to last
        create_symlink(chkpt_path, 'latest.pth')

        #symlink to best
        if is_best:
            create_symlink(chkpt_path, 'best.pth')

    def train(self, l_create_folder=None, dvc_metrics=False):
        settings = self.config["train"]["settings"]
        max_acc_eval = 0.
        epoch_time = AverageMeter()
        live = Live("artifacts/metrics_eval") if self.n_gpus < 2 and dvc_metrics else None

        print("Run ", settings["num_epochs"], " epochs * ", len(self.dl.dataset), " (", len(self.dl), " iters * ", self.config["train"]["dataloader"]["batch_size"], " input)")
        for epoch in range(self.epoch_start, settings["num_epochs"]):
            st = time.time()
            eloss = self._run_epoch(epoch)
            epoch_time.update(time.time() - st)
            self.dl.dataset.update_if_needed(self.head.get_losses(self.print_loss_details) if hasattr(self.head, 'get_losses') else None)

            if l_create_folder is not None:
                l_create_folder()
            max_acc_eval = self._eval_and_save(epoch, settings["eval_interval"], max_acc_eval, live, settings["save_interval"])

            if live is not None:
                live.log_metric("loss", eloss)
                live.next_step()

        if live is not None:
            live.end()

        print(f"Training time : {epoch_time.avg:.3f} ms by epoch")

    def _eval_and_save(self, epoch, eval_interval, max_acc_eval, live, save_interval):
        if self.device.index in (None, self.gpu_id_eval):
            if (epoch+1) % eval_interval == 0:
                is_best = False
                if self.dl_val is not None:
                    acc_eval = self._run_eval()
                    if acc_eval > max_acc_eval:
                        max_acc_eval = acc_eval
                        is_best = True

                    if is_best:
                        print("Is new best!")
                        if live is not None:
                            live.log_metric("top1", max_acc_eval)
                    if ((epoch+1) % save_interval == 0) or is_best:
                        self._save_checkpoint(epoch, is_best)
        return max_acc_eval