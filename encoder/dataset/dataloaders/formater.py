def split_batch(batch, device, gt_name):
############## INPUTS ##############
    inputs, labels = batch['keypoint'].to(device), batch[gt_name].to(device)
    if 'orientation' in batch:
        ori = batch['orientation'].to(device)
        inputs = (inputs, ori)
    if 'mask' in batch:
        masks = batch['mask'].to(device)
        inputs = (inputs, masks)
    if 'indexes_optim' in batch:
        indexes_optim = batch['indexes_optim'].to(device)
        inputs = (inputs, indexes_optim)

############## LABELS ##############
    if 'duration' in batch:
        duration = batch['duration']
        labels = (labels, duration)

    if 'keypoint_gt' in batch:
        labels = (labels, batch['keypoint_gt'])
    if 'cameras' in batch:
        cams =  batch['cameras'].to(device)
        if type(labels) is tuple:
            labels = labels + (cams, )
        else:
            labels = (labels, cams)

    if 'indexes' in batch:
            indexes =  batch['indexes'].to(device)
            if type(labels) is tuple:
                labels = labels + (indexes, )
            else:
                labels = (labels, indexes)

    if 'frame_dir' in batch:
        frame_dir = batch['frame_dir']
        if type(labels) is tuple:
            labels = labels + (frame_dir, )
        else:
            labels = (labels, frame_dir)

    return inputs, labels