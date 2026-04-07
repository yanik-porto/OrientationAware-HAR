import argparse
from torch.utils.data import DataLoader
from dvclive import Live
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm 

from encoder.encoders import *
from encoder.dataset.datasets import *
from heads import *
from encoder.dataset.tools.config import load_config
from encoder.dataset.tools.measure import AverageMeter, APMeter
from tools.evaluation import accuracy, map_gtidx_to_color, accuracy_multiple_labels, top_k_by_action, accuracy_text_embed
from tools.checkpoint import clean_checkpoint
from language import Language, filter_texts_with_map
from encoder.dataset.dataloaders.formater import split_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Test motion head")
    parser.add_argument("config", type=str, help="Path to configuration file")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument('--params', type=str, default=None, help='set if some parameters should be loaded from yaml')
    parser.add_argument('--save_metric', action="store_true", default=False, help="set if metric has to be saved in dvc")
    parser.add_argument('--name', type=str, default="testset", help="name of the dataset to load in config")
    parser.add_argument('--gpu_id_eval', type=int, default=0, help="Id of the gpu where running evaluation (and trining in case of single gpu)")
    parser.add_argument('--gt_name', type=str, default='label', help="Name of the gt variable to compare the output with")
    parser.add_argument('--force_classif', action="store_true", default=False, help="force the result to be done on classif output if double output")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config, args.params, update_default=False)

    # create model
    print("* create model ... ", end="")
    encoder = create_encoder(config)
    head = create_head(config)
    device = torch.device(args.gpu_id_eval)
    encoder, head = encoder.to(device), head.to(device)
    print("done")

    # load checkpoint
    print("* load checkpoint ...", end="")
    checkpoint = torch.load(args.checkpoint, map_location='cuda:' + str(args.gpu_id_eval))#, weights_only=False)
    if "epoch" in checkpoint:
        print("from epoch #", checkpoint["epoch"], " ... ", end="")
    chkpt_encoder = clean_checkpoint(checkpoint['encoder'], remove_fc=True)

    chkpt_head = clean_checkpoint(checkpoint['head'])
    encoder.load_state_dict(chkpt_encoder, strict=True)
    head.load_state_dict(chkpt_head, strict=True)
    print("done")

    # load testing dataset
    print("* create dataset ...", end="")
    dataset = create_dataset(config, setname=args.name)
    dl_settings = config["test"]["dataloader"]
    dl = DataLoader(dataset, **dl_settings)
    print(" done : ", len(dataset), " inputs")

    # test
    print("***************")
    print("**** test ****")
    print("Run ", len(dl), " iters")
    encoder.eval()
    head.eval()

    # create metrics
    metrics = {}
    if args.gt_name == "window_labels":
        metrics["Top1"] = APMeter()
    elif args.gt_name == "tad_labels":
        metrics["mAP@0.1"] = AverageMeter()
        metrics["mAP@0.25"] = AverageMeter()
        metrics["mAP@0.5"] = AverageMeter()
    else:
        metrics["Top1"] = AverageMeter()
        metrics["Top5"] = AverageMeter()
    top1_ml = AverageMeter()
    top5_ml = AverageMeter()
    all_outputs = []
    all_gts = []
    all_feats = []

    test_cross_modal = False        
    text_encoder = None
    language = None
    if 'encoder_text' in config:
        text_encoder = create_encoder(config, nodename='encoder_text')
        if 'text_encoder' in checkpoint:
            text_encoder.model.load_state_dict(checkpoint['text_encoder'], strict=True)
        text_encoder.to(device)
    if hasattr(head, 'forward_motion') and callable(head.forward_motion):
        if 'language' in config:
            language = Language(config, None, dl, text_encoder)
            text_embeds = None
            if language.mode != "json" or language.indexes_keyword_test.isdigit() :
                text_embeds = language.get_feats_test()
                text_embeds = text_embeds.to(device)
        elif text_encoder is not None:
            # TODO : check how to specify a certain list with angles
            texts = dl.dataset.label_map
            text_embeds = text_encoder(texts)
        else:
            text_embeds = torch.from_numpy(dl.dataset.text_embeds).to(device)

        text_feats = None
        if text_embeds is not None:
            text_feats = head.forward_text(text_embeds)
            text_feats = nn.functional.normalize(text_feats, dim=1)
        test_cross_modal = True

    num_clips = 1
    if "UniformSample" in config[args.name]["preprocessing"] and "num_clips" in config[args.name]["preprocessing"]["UniformSample"]:
        num_clips = config[args.name]["preprocessing"]["UniformSample"]["num_clips"]
    if "UniformSampleDecode" in config[args.name]["preprocessing"] and "num_clips" in config[args.name]["preprocessing"]["UniformSampleDecode"]:
        num_clips = config[args.name]["preprocessing"]["UniformSampleDecode"]["num_clips"]

    # iter on dataset
    for ii, batch in tqdm(enumerate(dl)):
        keypoints, labels = split_batch(batch, device, args.gt_name)

        if num_clips > 1:
            assert keypoints.shape[1]%num_clips==0, str(keypoints.shape) + " % " + str(num_clips)
            N, M, T, V, C = keypoints.shape
            keypoints = keypoints.reshape(N*num_clips, M//num_clips, T, V, C)

        if 'orientation' in batch: # TODO : handle this in split_batch
            ori = batch['orientation'].to(device)
            if num_clips > 1:
                ori = ori.repeat(num_clips, 1, 1)
                feats = encoder((keypoints[0], ori))
            else:
                feats = encoder(keypoints)
        else:
            feats = encoder(keypoints)

        if test_cross_modal:
            if 'texts' in batch:
                texts = batch['texts']
                texts = [text[0] for text in texts] # collate function convert list of string to list of list
                if dl.dataset.classes_map is not None:
                    texts = filter_texts_with_map(texts, dl.dataset.classes_map, dl.dataset.num_classes)
                text_embeds = text_encoder(texts)
                motion_feats_single, motion_feats, text_feats = head((feats, text_embeds))
                motion_logits = (motion_feats_single, motion_feats)
                text_feats = nn.functional.normalize(text_feats, dim=1)
            elif language is not None and text_feats is None:
                text_embeds = language.get_feats_test(batch)
                text_embeds = text_embeds.to(device)
                outs = head((feats, text_embeds))
                text_feats = outs[2]
                motion_logits = outs[0:2]
                if len(outs) > 3:
                    motion_logits = (*motion_logits, outs[3])
                text_feats = nn.functional.normalize(text_feats, dim=1)
            else:
                motion_logits = head.forward_motion(feats)
        else:
            motion_logits = head(feats)
            if num_clips > 1:
                if type(motion_logits) is tuple:
                    motion_logits = motion_logits[0]
                motion_logits = motion_logits.mean(0).unsqueeze(0) # TODO : handle case where batch_size > 1

        if test_cross_modal:
            if len(motion_logits) == 3 and args.force_classif:
                acc1, acc5 = accuracy(motion_logits[2], labels, topk=(1, 5))
                output = motion_logits[2]
            else:
                [acc1, acc5], output = accuracy_text_embed((motion_logits[0], motion_logits[1], text_feats), labels, (1, 5))
            metrics["Top1"].update(float(acc1[0].detach().cpu().numpy()), keypoints[0].size(0))
            metrics["Top5"].update(float(acc5[0].detach().cpu().numpy()), keypoints[0].size(0))
        else:
            if args.gt_name == "label":
                if type(motion_logits) is tuple:
                    motion_logits = motion_logits[0]
                if len(motion_logits.shape) == 3:
                    M = motion_logits.shape[1]
                    motion_logits = motion_logits.flatten(0, 1)
                    labels = labels.repeat_interleave(M)
                acc1, acc5 = accuracy(motion_logits, labels, topk=(1, 5))

            metrics["Top1"].update(acc1[0].detach().cpu().numpy(), len(batch['keypoint']))
            metrics["Top5"].update(acc5[0].detach().cpu().numpy(), len(batch['keypoint']))

            output = motion_logits

        if 'binary_labels' in batch:
            acc1_ml, acc5_ml = accuracy_multiple_labels(output.detach().cpu(), batch['binary_labels'], topk=(1, 5))
            top1_ml.update(acc1_ml, len(batch['keypoint']))
            top5_ml.update(acc5_ml, len(batch['keypoint']))

        if type(output) is tuple:
            output = output[0]
        if type(labels) is tuple:
            labels = labels[0]

        if args.gt_name not in ("tad_labels", "window_labels"):
            all_outputs.extend(output.detach().cpu().numpy())
            all_gts.extend(labels.detach().cpu().numpy())

        if args.save_metric:
            if test_cross_modal:
                feats_head = nn.functional.normalize(motion_logits[0], dim=1)
                feats_head = feats_head.detach().cpu().numpy()
                all_feats.append(feats_head[0].reshape(-1))
            else:
                feats = feats.detach().cpu().numpy()
                all_feats.append(feats[0, 0:1].reshape(-1))

            if len(all_feats) >= 5000:
                break

        output = None


    for metric_name, metric_value in metrics.items():
        print(f" | {metric_name}: {metric_value.value():.2f}", end="")
    print("")
    if top1_ml.count > 0:
        print("Accuracy Top1 Multi Labels : ", top1_ml.value())
        print("Accuracy Top5 Multi Labels : ", top5_ml.value())

    if args.gt_name not in ("tad_labels", "window_labels"):
        np.set_printoptions(legacy='1.25')
        top1_by_action  = top_k_by_action(np.array(all_outputs), all_gts)
        print(dict(sorted(top1_by_action.items())).values())
        # print(dict(sorted(top1_by_action.items())))

    if args.save_metric:
        with Live("artifacts/metrics_" + args.name) as live:
            for metric_name, metric_value in metrics.items():
                live.log_metric(metric_name, metric_value.avg)

            # build confusion matrix
            pred = list(np.argmax(np.array(all_outputs), axis=1))
            all_gt_idxs = all_gts
            if max(pred) < len(dataset.label_map) and max(all_gts) < len(dataset.label_map):
                pred = [dataset.label_map[p] for p in pred]
                all_gts = [dataset.label_map[l] for l in all_gts]
            live.log_sklearn_plot(
                "confusion_matrix", all_gts, pred, name="cm.json")

            # build tsne
            if not test_cross_modal:
                tsne = TSNE(n_components=2, random_state=42)
                X_tsne = tsne.fit_transform(np.array(all_feats))
                fig, ax = plt.subplots(1,1)
                all_color_idxs, colorlabels = map_gtidx_to_color(all_gt_idxs[:len(all_feats)], dataset.label_map)
                scatter_motion = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=all_color_idxs, marker='.', cmap='gist_rainbow', linewidths=0.5)
                handles, _ = scatter_motion.legend_elements(prop='colors')
                plt.legend(handles, colorlabels)

            else:
                print("motion : ", torch.mean(torch.tensor(all_feats)).item(), ' mean | ', torch.var(torch.tensor(all_feats)).item(), ' var')
                print("text : ", torch.mean(torch.tensor(text_feats)).item(), ' mean | ', torch.var(torch.tensor(text_feats)).item(), ' var')
                tsne = TSNE(n_components=2, random_state=42)
                X_all = np.concatenate((all_feats, text_feats.detach().cpu().numpy()))
                X_tsne = tsne.fit_transform(np.array(X_all))
                fig, ax = plt.subplots(1,1)
                all_color_idxs, colorlabels = map_gtidx_to_color(all_gt_idxs[:len(all_feats)], dataset.label_map)
                text_color_idxs, colorlabels = map_gtidx_to_color(range(len(text_feats)), dataset.label_map)

                motion_tsne = X_tsne[:len(all_feats)]
                text_tsne = X_tsne[len(all_feats):]

                scatter_motion = plt.scatter(motion_tsne[:,0], motion_tsne[:,1], c=all_color_idxs, marker='.', cmap='gist_rainbow', linewidths=0.5)
                scatter_text = plt.scatter(text_tsne[:,0], text_tsne[:,1], c=text_color_idxs, marker='D', cmap='gist_rainbow', linewidths=4.0, edgecolors='black')

                handles, _ = scatter_text.legend_elements(prop='colors')
                plt.legend(handles, colorlabels)

            plt.legend()
            live.log_image("tsne.png", fig)
