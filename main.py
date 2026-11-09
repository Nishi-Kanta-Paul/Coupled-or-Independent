import os
import argparse
import datetime
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
for _key, _sub in (('HF_HOME', 'hf'), ('TORCH_HOME', 'torch'), ('XDG_CACHE_HOME', 'xdg'), ('MPLCONFIGDIR', 'mpl'),
                   ('EASYOCR_MODULE_PATH', 'easyocr')):
    os.environ.setdefault(_key, str(CODE_ROOT / '.cache' / _sub))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch

from util import *
from models import *


STAGES = ['split', 'segment', 'encode', 'channel', 'factorial', 'tools', 'gate', 'agent', 'evaluate', 'report']


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Coupled or Independent? fault coupling in agentic medical vision systems', add_help=False)
    parser.add_argument('--stage', default='all', choices=['all'] + STAGES,
                        help='pipeline stage to run (default: all, in order)')
    parser.add_argument('--data-path', default=str(CODE_ROOT.parent / 'PAD-UFES-20'), type=str,
                        help='PAD-UFES-20 root containing metadata.csv and images/')
    parser.add_argument('--hiba-path', default='', type=str,
                        help='HIBA (ISIC collection 251) root with images and ISIC metadata CSV; empty to skip external evaluation')
    parser.add_argument('--output_dir', default=str(CODE_ROOT / 'results' / 'full'), type=str,
                        help='path where all intermediate and final outputs are stored')
    parser.add_argument('--smoke-image', default='', type=str,
                        help='run the whole workflow on this single PAD-UFES-20 image id as an executability check')
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--overwrite', action='store_true', help='recompute cached outputs of the selected stages')

    parser.add_argument('--split-fractions', default=[0.70, 0.15, 0.15], type=float, nargs=3,
                        help='patient-level fractions for reference pool, validation and internal test')
    parser.add_argument('--no-stratify', action='store_true', help='do not stratify the patient split by cancer status')

    parser.add_argument('--sam-model', default='facebook/sam-vit-huge', type=str, help='SAM checkpoint (ViT-H)')
    parser.add_argument('--crop-margin', default=1.25, type=float, help='lesion crop margin around the mask bounding box')

    parser.add_argument('--font-path', default='', type=str, help='TrueType font for payload rendering (default: DejaVu Sans)')
    parser.add_argument('--payload-width-frac', default=0.9, type=float,
                        help='lesion payload width as a fraction of the lesion bounding-box width')
    parser.add_argument('--low-contrast-delta', default=0.12, type=float,
                        help='luminance offset of low-contrast payload text relative to the local background (0-1)')
    parser.add_argument('--line-gap', default=0.2, type=float, help='gap between wrapped payload lines relative to line height')
    parser.add_argument('--payload-index', default=-1, type=int,
                        help='payload phrase index per direction; -1 assigns one phrase per image by a fixed hash')
    parser.add_argument('--blur-severity', default=2, type=int, help='ImageNet-C Gaussian blur severity')
    parser.add_argument('--jpeg-severity', default=4, type=int, help='ImageNet-C JPEG compression severity')
    parser.add_argument('--corruption-scale', default='relative', choices=['relative', 'native'],
                        help='relative scales the ImageNet-C blur sigma from 224 px to the image short side')
    parser.add_argument('--render-heights', default=[3, 4, 6, 8, 10, 14, 20], type=float, nargs='+',
                        help='payload heights at the encoder input for the render-scale sweep (px), as tested in Fig. 2b')
    parser.add_argument('--provenance-image', default='PAT_433_852_20', type=str,
                        help='image whose payload scale under both orders is marked in the render-scale figure (Fig. 1 provenance)')
    parser.add_argument('--gate-figure-order', default='crop', choices=['crop', 'full'],
                        help='pipeline order shown in the gate flag-rate figure')

    parser.add_argument('--channel-encoders', default=['biomedclip', 'pubmedclip', 'clip'], nargs='+',
                        choices=list(ENCODER_REGISTRY.keys()), help='contrastive encoders for channel-level analysis')
    parser.add_argument('--agent-encoder', default='auto', choices=['auto'] + list(ENCODER_REGISTRY.keys()),
                        help='encoder used by the agent; auto selects the best clean validation retrieval precision')
    parser.add_argument('--encoder-resolution', default=224, type=int, help='encoder input resolution used to report payload scale')
    parser.add_argument('--encode-batch-size', default=32, type=int)
    parser.add_argument('--knn', default=5, type=int, help='number of retrieved neighbours k_nn')
    parser.add_argument('--factorial-heights', default=None, type=float, nargs=2, metavar=('SMALL', 'LARGE'),
                        help='payload heights at the encoder input (px) for the scale x cropping factorial; default: median '
                             'full-image and crop-first heights of the injected internal test images')
    parser.add_argument('--factorial-encoder', default='auto', choices=['auto'] + list(ENCODER_REGISTRY.keys()),
                        help='encoder for the factorial; auto uses the agent encoder')
    parser.add_argument('--capture-view', default='crop', choices=['crop', 'full'],
                        help='encoder view summarised in the directional-capture figure and controls table')

    parser.add_argument('--reasoners', default=['Qwen/Qwen2.5-VL-7B-Instruct', 'microsoft/llava-med-v1.5-mistral-7b'], nargs='+',
                        help='frozen vision-language reasoners')
    parser.add_argument('--reasoner-dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--reasoner-image-max-side', default=0, type=int,
                        help='downscale images for the reasoner to this longest side; 0 keeps the native resolution')
    parser.add_argument('--arms', default=list(ARMS), nargs='+', choices=list(ARMS),
                        help='single-shot arm, the two agentic orders and the two retrieval-ablation arms')
    parser.add_argument('--decodes', default=5, type=int, help='randomised decodes k per case')
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--top-p', default=0.95, type=float)
    parser.add_argument('--max-new-tokens', default=1024, type=int)
    parser.add_argument('--react-max-steps', default=6, type=int, help='maximum ReAct turns before a final answer is requested')
    parser.add_argument('--tau', default=0.8, type=float, help='stability threshold for confident-wrong answers')

    parser.add_argument('--gate-work-side', default=1024, type=int, help='longest side at which gate statistics are computed')
    parser.add_argument('--gate-target-rate', default=0.05, type=float, help='clean validation flag-rate target')
    parser.add_argument('--text-sigma', default=4.0, type=float, help='local normalisation scale of the text score')
    parser.add_argument('--text-threshold', default=2.0, type=float, help='locally normalised response threshold of the text score')
    parser.add_argument('--text-close', default=9, type=int, help='horizontal closing length of the text score')
    parser.add_argument('--text-open', default=31, type=int, help='horizontal opening length of the text score')
    parser.add_argument('--contrast-window', default=15, type=int, help='window of the local-contrast map')
    parser.add_argument('--craft-canvas', default=0, type=int, help='CRAFT canvas size; 0 uses the full image resolution')
    parser.add_argument('--coverage-targets', default=[1.00, 0.90, 0.88, 0.82, 0.70], type=float, nargs='+')

    parser.add_argument('--n-boot', default=1000, type=int, help='patient-clustered bootstrap resamples')
    parser.add_argument('--ci', default=95.0, type=float, help='bootstrap interval level')
    parser.add_argument('--ece-bins', default=10, type=int)
    parser.add_argument('--mask-iou-subset', default=100, type=int, help='random images per corpus for the mask controls')
    return parser


def main(args):
    print(args)
    torch.manual_seed(args.seed)
    ws = Workspace(args.output_dir)
    runners = {
        'split': stage_split, 'segment': stage_segment, 'encode': stage_encode, 'channel': stage_channel,
        'factorial': stage_factorial,
        'tools': stage_tools, 'gate': stage_gate, 'agent': stage_agent, 'evaluate': stage_evaluate, 'report': stage_report,
    }
    stages = STAGES if args.stage == 'all' else [args.stage]
    start_time = time.time()
    for stage in stages:
        stage_start = time.time()
        utils.log(f'stage {stage}: start')
        runners[stage](args, ws)
        utils.log(f'stage {stage}: done in {datetime.timedelta(seconds=int(time.time() - stage_start))}')
    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('Total time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Coupled or Independent? fault coupling in agentic medical vision systems', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
