import torch
# from configs.coco.ViTPose_base_coco_256x192 import model
from .builder.heads.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead
# import TopdownHeatmapSimpleHead
from .builder.backbones import ViT
# print(model)
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module

models = {
    "ViTPose_base_coco_256x192": dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=17,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict()),
    
"ViTPose_base_simple_coco_256x192":  dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=768,
        num_deconv_layers=0,
        num_deconv_filters=[],
        num_deconv_kernels=[],
        upsample=4,
        extra=dict(final_conv_kernel=3, ),
        out_channels=17,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type='GaussianHeatmap',
        modulate_kernel=11,
        use_udp=True)),
"ViTPose_base_wholebody": dict(
        type='TopDown',
    pretrained=None,
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=133,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))



    }

def split_vit_plus(ckpt):
    import copy

    
    experts = dict()

    new_ckpt = copy.deepcopy(ckpt)

    state_dict = new_ckpt['state_dict']

    for key, value in state_dict.items():
        if 'mlp.experts' in key:
            experts[key] = value

    keys = ckpt['state_dict'].keys()

    target_expert = 0
    new_ckpt = copy.deepcopy(ckpt)

    for key in keys:
        if 'mlp.fc2' in key:
            value = new_ckpt['state_dict'][key]
            value = torch.cat([value, experts[key.replace('fc2.', f'experts.{target_expert}.')]], dim=0)
            new_ckpt['state_dict'][key] = value

    names = ['aic', 'mpii', 'ap10k', 'apt36k','wholebody']
    num_keypoints = [14, 16, 17, 17, 133]
    weight_names = ['keypoint_head.deconv_layers.0.weight', 
                    'keypoint_head.deconv_layers.1.weight', 
                    'keypoint_head.deconv_layers.1.bias', 
                    'keypoint_head.deconv_layers.1.running_mean', 
                    'keypoint_head.deconv_layers.1.running_var', 
                    'keypoint_head.deconv_layers.1.num_batches_tracked', 
                    'keypoint_head.deconv_layers.3.weight', 
                    'keypoint_head.deconv_layers.4.weight', 
                    'keypoint_head.deconv_layers.4.bias', 
                    'keypoint_head.deconv_layers.4.running_mean', 
                    'keypoint_head.deconv_layers.4.running_var', 
                    'keypoint_head.deconv_layers.4.num_batches_tracked', 
                    'keypoint_head.final_layer.weight', 
                    'keypoint_head.final_layer.bias']
    
    exist_range = True


    new_ckpt = copy.deepcopy(ckpt)
    i=4
    target_expert = i + 1

    for key in keys:
        if 'mlp.fc2' in key:
            expert_key = key.replace('fc2.', f'experts.{target_expert}.')
            if expert_key in experts:
                value = new_ckpt['state_dict'][key]
                value = torch.cat([value, experts[expert_key]], dim=0)
            else:
                exist_range = False

            new_ckpt['state_dict'][key] = value

    for tensor_name in weight_names:
        new_ckpt['state_dict'][tensor_name] = new_ckpt['state_dict'][tensor_name.replace('keypoint_head', f'associate_keypoint_heads.{i}')]

    for tensor_name in ['keypoint_head.final_layer.weight', 'keypoint_head.final_layer.bias']:
        new_ckpt['state_dict'][tensor_name] = new_ckpt['state_dict'][tensor_name][:num_keypoints[i]]
    
    # remove unnecessary part in the state dict
    for j in range(5):
        # remove associate part
        for tensor_name in weight_names:
            new_ckpt['state_dict'].pop(tensor_name.replace('keypoint_head', f'associate_keypoint_heads.{j}'))
    # remove expert part
    keys = new_ckpt['state_dict'].keys()
    for key in list(keys):
        if 'expert' in key:
            new_ckpt['state_dict'].pop(key)
    return new_ckpt  
        # torch.save(new_ckpt, os.path.join(args.target, f'{names[i]}.pth'))
def build_model(model_name,checkpoint=None):
    try:
        # path = 'builder.configs.coco.'+model_name
        # mod = import_module(
        #     path
        # )
        
        # model = getattr(mod, "model")
        model = models[model_name]
        # from path import model
    except:
        raise ValueError('not a correct config')

        
    head = TopdownHeatmapSimpleHead(in_channels=model['keypoint_head']['in_channels'], 
                                    out_channels=model['keypoint_head']['out_channels'],
                                    num_deconv_filters=model['keypoint_head']['num_deconv_filters'],
                                    num_deconv_kernels=model['keypoint_head']['num_deconv_kernels'],
                                    num_deconv_layers=model['keypoint_head']['num_deconv_layers'],
                                    extra=model['keypoint_head']['extra'])
    # print(head)
    backbone = ViT(img_size=model['backbone']['img_size'],
                patch_size=model['backbone']['patch_size']
                ,embed_dim=model['backbone']['embed_dim'],
                depth=model['backbone']['depth'],
                num_heads=model['backbone']['num_heads'],
                ratio = model['backbone']['ratio'],
                mlp_ratio=model['backbone']['mlp_ratio'],
                qkv_bias=model['backbone']['qkv_bias'],
                drop_path_rate=model['backbone']['drop_path_rate']
                )

    class VitPoseModel(nn.Module):
        def __init__(self,backbone,keypoint_head):
            super(VitPoseModel, self).__init__()
            self.backbone = backbone
            self.keypoint_head = keypoint_head
        def forward(self,x):
            x = self.backbone(x)
            x = self.keypoint_head(x)
            return x
    
    pose = VitPoseModel(backbone, head)
    if checkpoint is not None:
        check = torch.load(checkpoint)
        if 'wholebody' in model_name:
            check = split_vit_plus(check)
        pose.load_state_dict(check['state_dict'])
    return pose
# pose = build_model('ViTPose_base_coco_256x192','./models/vitpose-b-multi-coco.pth')

