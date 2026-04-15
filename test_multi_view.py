import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dvclive import Live

from encoder.encoders import *
from encoder.dataset.datasets import *
from heads import *
from encoder.dataset.tools.config import load_config
from encoder.dataset.tools.measure import AverageMeter
from tools.checkpoint import clean_checkpoint, greedy_soup
from tools.evaluation import accuracy, top_k_by_action, accuracy_text_embed, map_gtidx_to_color
from language import Language


def parse_args():
    parser = argparse.ArgumentParser(description="Test motion head")
    parser.add_argument("config", type=str, help="Path to configuration file")
    parser.add_argument("checkpoints_folder", type=str, help="Path to checkpoint files")
    parser.add_argument('--params', type=str, default=None, help='set if some parameters should be loaded from yaml')
    parser.add_argument('--save_metric', action="store_true", default=False, help="set if metric has to be saved in dvc")
    parser.add_argument('--name', type=str, default="testset", help="name of the dataset to load in config")
    parser.add_argument('--default_angle_degrees', action='store_true', default=False, help="do not get angle_degrees from preprocessing")
    parser.add_argument('--gpu_id_eval', type=int, default=0, help="Id of the gpu where running evaluation (and trining in case of single gpu)")
    parser.add_argument('--indexes_keyword', type=str, default='all', help="keyword to choose the corresponding model")
    parser.add_argument('--force_classif', action="store_true", default=False, help="force the result to be done on classif output if double output")
    return parser.parse_args()


def get_model_from_index(models, vi, batch, keyword):
    if keyword == 'all':
        return models[vi], vi
    elif keyword.isdigit():
        return models[int(keyword)], int(keyword)
    else:
        assert keyword in  batch, batch.keys()
        idx = int(batch[keyword][0])
        return models[idx], idx

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config, args.params, update_default=False)

    # create model
    print("* create model ... ", end="")
    encoder = create_encoder(config)
    head = create_head(config)
    device = torch.device(args.gpu_id_eval)
    encoder, head = encoder.to(device), head.to(device)
    encoder.eval()
    head.eval()
    print("done")

    if args.default_angle_degrees or "ProjectToGtCamera" not in config[args.name]["preprocessing"]:
        angle_degrees = list(range(-180, 180, 30))
    else:
        angle_degrees = config[args.name]["preprocessing"]["ProjectToGtCamera"]["angle_degrees"]
    print(angle_degrees)

    # load checkpoint
    models = []
    text_encoder = None
    if True:
        checkpoint_path = args.checkpoints_folder
        checkpoint = torch.load(checkpoint_path, map_location='cuda:' + str(args.gpu_id_eval))
        print("load checkpoint from epoch #", checkpoint["epoch"], " ... ", end="")
        chkpt_encoder = clean_checkpoint(checkpoint['encoder'], remove_fc=True)
        chkpt_head = clean_checkpoint(checkpoint['head'])
        encoder.load_state_dict(chkpt_encoder, strict=True)
        head.load_state_dict(chkpt_head, strict=True)
        encoder.eval()
        head.eval()
        model = [encoder, head]
        if 'text_encoder' in checkpoint:
            text_encoder = create_encoder(config, nodename='encoder_text')
            text_encoder.model.load_state_dict(checkpoint['text_encoder'], strict=True)
            text_encoder.eval()
            text_encoder.to(device)
            model.append(text_encoder)
        for angle in angle_degrees:
            models.append(model)
        print("done")

    # load testing dataset
    print("* create dataset ...", end="")
    dataset = create_dataset(config, setname=args.name)
    dl_settings = config["test"]["dataloader"]
    dl = DataLoader(dataset, **dl_settings)
    print("done")

    n_persons = 1
    if "FormatGCNInput" in config[args.name]["preprocessing"]:
        n_persons = config[args.name]["preprocessing"]["FormatGCNInput"]["num_person"]

    # test
    print("***************")
    print("**** test ****")
    print("Run ", len(dl), " iters")
    resByAngle = {}
    all_outputs = []
    all_gts = []
    top1 = AverageMeter()
    top5 = AverageMeter()
    if args.save_metric:
        all_feats = [[] for _ in range(len(angle_degrees))]
        all_text_feats = []

    test_cross_modal = False
    language = None
    if hasattr(head, 'forward_motion') and callable(head.forward_motion):
        if 'language' in config:
            language = Language(config, None, dl, text_encoder)
            if language.mode == 'embeds':
                text_embeds = language.get_feats_test().to(device)
                text_feats = []
                for it, model in enumerate(models):
                    text_feat = model[1].forward_text(text_embeds)
                    text_feat = nn.functional.normalize(text_feat, dim=1)
                    text_feats.append(text_feat)

        elif not 'encoder_text' in config:
            text_embeds = torch.from_numpy(dl.dataset.text_embeds).to(device)
            text_feats = []
            for it, model in enumerate(models):
                text_feat = model[1].forward_text(text_embeds)
                text_feat = nn.functional.normalize(text_feat, dim=1)
                text_feats.append(text_feat)
        test_cross_modal = True

    for ii, batch in tqdm(enumerate(dl)):
        keypoints, labels = batch['keypoint'].to(device), batch['label'].to(device)

        if test_cross_modal and 'encoder_text' in config:
            text_feats = []

        outputs = []
        for vi in range(0, keypoints.shape[1], n_persons):
            view_keypoints = keypoints[:, vi:vi+n_persons, ...] # range to keep dimension 1
            model, model_idx = get_model_from_index(models, vi, batch, args.indexes_keyword)
            if 'orientation' in batch:
                ori = batch['orientation'].to(device)
                if ori.shape[1] > 1:
                    ori = ori[:, vi:vi+n_persons, :]
                view_keypoints = (view_keypoints, ori)

            feats = model[0](view_keypoints)
            if test_cross_modal:
                if len(model) == 3:
                    if language is not None:
                        if language.mode == "json" and language.indexes_keyword_test == "indexes":
                            language.indexes_keyword_test = str(model_idx)
                        texts = language.get_texts_test(batch)
                    elif 'texts' in batch:
                        texts = batch['texts']
                        if len(texts[0]) > 1: # case where all texts viewpoints are passed
                            texts = texts[model_idx]
                        texts = [text[0] for text in texts]
                    else:
                        texts = dl.dataset.label_map
                    text_embeds = model[2](texts)

                    outs = model[1]((feats, text_embeds))

                    view_output = outs[0:2]
                    text_feat = outs[2]
                    if len(outs) > 3:
                        view_output = (*view_output, outs[3])

                    text_feats.append(text_feat)

                    if args.save_metric:
                        motion_feats_single = view_output[0]
                        m_feats = nn.functional.normalize(motion_feats_single[0, :], dim=0)
                        all_feats[model_idx].append(m_feats.detach().cpu().numpy())
                else:
                    view_output = model[1].forward_motion(feats)
            else:
                view_output = model[1](feats)

            if type(view_output) is tuple:
                if len(view_output) > 2 and args.force_classif:
                    view_output = view_output[2]
                else:
                    view_output = view_output[0]

            outputs.append(view_output)
        output = torch.stack(outputs, dim=1)


        if test_cross_modal:
            if args.force_classif:
                output = output.mean(1)
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            else:
                [acc1, acc5], output = accuracy_text_embed((output, output, text_feats), labels, (1, 5))

            if args.save_metric:
                if ii == 0:
                    for tf in text_feats:
                        tf = nn.functional.normalize(tf, dim=1)
                        all_text_feats.append(tf.detach().cpu().numpy())

        else:
            output = output.mean(1)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0].detach().cpu().numpy(), len(batch['keypoint']))
        top5.update(acc5[0].detach().cpu().numpy(), len(batch['keypoint']))

        all_outputs.extend(output.detach().cpu().numpy())
        all_gts.extend(labels.detach().cpu().numpy())

        if args.save_metric and ii >= 99:
            break

    print("Accuracy Top1 : ", top1.avg)
    print("Accuracy Top5 : ", top5.avg)

    np.set_printoptions(legacy='1.25')
    top1_by_action  = top_k_by_action(np.array(all_outputs), all_gts)
    print(dict(sorted(top1_by_action.items())).values())

    if args.save_metric:
        if test_cross_modal:
            with Live("artifacts/metrics_" + args.name) as live:
                all_gt_idxs = all_gts
                tsne = TSNE(n_components=2, random_state=42)

                # reshape features from V, N to V*N
                all_feats = np.array(all_feats)
                Vm, Nm, Cm = all_feats.shape
                all_feats = np.reshape(all_feats, (Vm*Nm, Cm))

                all_text_feats = np.array(all_text_feats)
                Vt, Lt, Ct = all_text_feats.shape
                all_text_feats = np.reshape(all_text_feats, (Vt*Lt, Ct))

                all_color_idxs, colorlabels = map_gtidx_to_color(all_gt_idxs[:Nm], dataset.label_map)
                text_color_idxs, colorlabels = map_gtidx_to_color(range(Lt), dataset.label_map)

                angle_indexes = [0, 3, 6, 9]
                markers = ['1', '2', '3', '4']
                markers_text = ['v', '<', '^', '>']

                # TSNE all
                X_all = np.concatenate((all_feats, all_text_feats))
                X_tsne = tsne.fit_transform(np.array(X_all))
                fig, ax = plt.subplots(1,1)

                motion_tsne = X_tsne[:Vm*Nm]
                text_tsne = X_tsne[Vm*Nm:]
                motion_tsne = np.reshape(motion_tsne, (Vm, Nm, -1))
                text_tsne = np.reshape(text_tsne, (Vt, Lt, -1))

                for im, iv in enumerate(angle_indexes):
                    scatter_motion = plt.scatter(motion_tsne[iv, :, 0], motion_tsne[iv, :, 1], c=all_color_idxs, marker=markers[im], cmap='gist_rainbow', linewidths=0.5)
                    scatter_text = plt.scatter(text_tsne[iv, :, 0], text_tsne[iv, :, 1], c=text_color_idxs, marker=markers_text[im], cmap='gist_rainbow', linewidths=2.0)#, edgecolors='black')
                    handles, _ = scatter_text.legend_elements(prop='colors')
                plt.legend(handles, colorlabels)
                live.log_image("tsne.png", fig)

                # TSNE Text
                text_alone_tsne = tsne.fit_transform(np.array(all_text_feats))
                fig_text, ax = plt.subplots(1,1)
                text_alone_tsne = np.reshape(text_alone_tsne, (Vt, Lt, -1))
                for im, iv in enumerate(angle_indexes):
                    scatter_text_alone = plt.scatter(text_alone_tsne[iv, :, 0], text_alone_tsne[iv, :, 1], c=text_color_idxs, marker=markers_text[im], cmap='gist_rainbow', linewidths=2.0)#, edgecolors='black')
                    handles_text, _ = scatter_text_alone.legend_elements(prop='colors')
                plt.legend(handles_text, colorlabels)
                live.log_image("tsne_text.png", fig_text)

                # TSNE Motion
                motion_alone_tsne = tsne.fit_transform(np.array(all_feats))
                fig_motion, ax = plt.subplots(1,1)
                motion_alone_tsne = np.reshape(motion_alone_tsne, (Vm, Nm, -1))
                for im, iv in enumerate(angle_indexes):
                    scatter_motion_alone = plt.scatter(motion_alone_tsne[iv, :, 0], motion_alone_tsne[iv, :, 1], c=all_color_idxs, marker=markers[im], cmap='gist_rainbow', linewidths=0.5)#, edgecolors='black')
                    handles_motion, _ = scatter_motion_alone.legend_elements(prop='colors')
                plt.legend(handles_motion, colorlabels)
                live.log_image("tsne_motion.png", fig_motion)

                # TSNE / angle
                all_feats = np.reshape(all_feats, (Vm, Nm, Cm))
                all_text_feats = np.reshape(all_text_feats, (Vt, Lt, Ct))
                for im, iv in enumerate(angle_indexes):
                    X_merge = np.concatenate((all_feats[iv], all_text_feats[iv]))
                    X_tsne = tsne.fit_transform(np.array(X_merge))
                    motion_tsne = X_tsne[:Nm]
                    text_tsne = X_tsne[Nm:]
                    fig, ax = plt.subplots(1,1)
                    scatter_motion = plt.scatter(motion_tsne[:, 0], motion_tsne[:, 1], c=all_color_idxs, marker=markers[im], cmap='gist_rainbow', linewidths=0.5)
                    scatter_text = plt.scatter(text_tsne[:, 0], text_tsne[:, 1], c=text_color_idxs, marker=markers_text[im], cmap='gist_rainbow', linewidths=2.0)#, edgecolors='black')
                    handles, _ = scatter_text.legend_elements(prop='colors')
                    plt.legend(handles, colorlabels)
                    live.log_image("tsne_" + str(iv) + ".png", fig)