from .pose_related import *
from .formatting import *
from .utils import *
from .sampling import *
from .joints_related import *
from .noise import *

def create_preprocessing(config):
    assert 'preprocessing' in config, "no preprocessing specified in config"

    cfg_preprocessing = config['preprocessing']

    steps = []

    for step in cfg_preprocessing:
        if step == "ToTensor":
            steps.append(ToTensor(**cfg_preprocessing[step]))
        elif step == "Collect":
            steps.append(Collect(**cfg_preprocessing[step]))
        elif step == "PreNormalize2D":
            steps.append(PreNormalize2D(**cfg_preprocessing[step]))
        elif step == "GenSkeFeat":
            steps.append(GenSkeFeat())
        elif step == "PoseDecode":
            steps.append(PoseDecode())
        elif step == "FormatGCNInput":
            steps.append(FormatGCNInput(**cfg_preprocessing[step]))
        elif step == "FormatGCNInputMV":
            steps.append(FormatGCNInputMV(**cfg_preprocessing[step]))
        elif step == "UniformSample":
            steps.append(UniformSample(**cfg_preprocessing[step]))
        elif step == "UniformSampleDecode":
            steps.append(UniformSampleDecode(**cfg_preprocessing[step]))
        elif step == "Resample":
            steps.append(Resample(**cfg_preprocessing[step]))
        elif step == "SampleFixedLength":
            steps.append(SampleFixedLength(**cfg_preprocessing[step]))
        elif step == "ProjectToDefinedCams":
            steps.append(ProjectToDefinedCams(**cfg_preprocessing[step]))
        elif step == "ProjectToRandomCamera":
            steps.append(ProjectToRandomCamera(**cfg_preprocessing[step]))
        elif step == "ProjectToGtCamera":
            steps.append(ProjectToGtCamera(**cfg_preprocessing[step]))
        elif step == "ProjectToClosestCamera":
            steps.append(ProjectToClosestCamera(**cfg_preprocessing[step]))
        elif step == "ProjectToSampledCams":
            steps.append(ProjectToSampledCams(**cfg_preprocessing[step]))
        elif step == "ProjectToRandomSampledCams":
            steps.append(ProjectToRandomSampledCams(**cfg_preprocessing[step]))
        elif step == "JointsToKeypoints":
            steps.append(JointsToKeypoints(**cfg_preprocessing[step]))
        elif step == "AddNoiseToKeypoints":
            steps.append(AddNoiseToKeypoints(**cfg_preprocessing[step]))
        elif step == "PadTime":
            steps.append(PadTime(**cfg_preprocessing[step]))
        elif step == "InverseAxis":
            steps.append(InverseAxis(**cfg_preprocessing[step]))
        elif step == "Centerize":
            steps.append(Centerize(**cfg_preprocessing[step]))
        elif step == "CenterizeJoints":
            steps.append(CenterizeJoints(**cfg_preprocessing[step]))
        elif step == "MaskViews":
            steps.append(MaskViews())
        elif step == "MaskViewsRandom":
            steps.append(MaskViewsRandom(**cfg_preprocessing[step]))
        elif step == "Normalize":
            steps.append(Normalize(**cfg_preprocessing[step]))
        elif step == "KeepIndexes":
            steps.append(KeepIndexes(**cfg_preprocessing[step]))
        elif step == "SampleDynamicIndexes":
            steps.append(SampleDynamicIndexes(**cfg_preprocessing[step]))
        elif step =="AppendToKeypoint":
            steps.append(AppendToKeypoint(**cfg_preprocessing[step]))
        elif step == "Replace":
            steps.append(Replace(**cfg_preprocessing[step]))
        else:
            print(step, " not handled yet in preprocessors")

    return Compose(steps)
