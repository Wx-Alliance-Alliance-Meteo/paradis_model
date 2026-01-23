import torch 
import torch.nn as nn
from   torch.nn.modules.utils import _pair
from   torch.nn               import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from   scipy                  import ndimage

import ml_collections
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import copy
import logging
import math
from os.path import join as pjoin
import numpy as np

ATTENTION_Q    = "MultiHeadDotProductAttention_1/query"
ATTENTION_K    = "MultiHeadDotProductAttention_1/key"
ATTENTION_V    = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT  = "MultiHeadDotProductAttention_1/out"
FC_0           = "MlpBlock_3/Dense_0"
FC_1           = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM       = "LayerNorm_2"

def get_cvh_Ix16_config():
    """"""
    config                        = ml_collections.ConfigDict()
    config.patches                = ml_collections.ConfigDict({'grid': (1, 1)})
    config.transformer            = ml_collections.ConfigDict()
    config.hidden_output_trans    = 512  #672
    config.transformer.mlp_dim    = 1024 #2048
    config.transformer.num_heads  = 16
    config.transformer.num_layers = 1
    config.head_channels          = 512 #672
    config.input_size             = (32,64,512)
    config.input_channel_scale1   = config.input_size[2]
    config.p_size                 = 8 #16
    config.patch_size             = (config.p_size,config.p_size) 
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate           = 0.1
    #config.classifier                         = None
    config.representation_size                = None
    config.resnet_pretrained_path             = None
    config.pretrained_path                    = None    
    # config.n_scales                           = int(np.log2(config.p_size) + 1)
    # config.n_skip                             = config.n_scales - 2 
    # config.scales                             = [i for i in range(config.n_scales)]


    config.n_scales                           = int(np.log2(config.p_size))
    config.n_skip                             = config.n_scales
    config.scales                             = [i for i in range(1,config.n_scales)]


    #################################################################
    starting_length        = config.input_size[0]
    starting_heigth        = config.input_size[1]
    starting_input_channel = config.input_size[2]
    Deph                   = config.head_channels
    LDS_Transformer        = config.hidden_output_trans
    patch_size             = config.p_size

    Final_length           = starting_heigth/patch_size
    Final_heigth           = starting_length/patch_size
    
    downward_in_channels   = {1:config.input_channel_scale1, 2:config.input_channel_scale1, 3:config.input_channel_scale1, 4:config.input_channel_scale1, 5:config.input_channel_scale1}

    config.downward_dim          = {'in': {},'out': {}}   
    config.upward_dim            = {'in': {},'out': {}}
    config.skip_dim              = {}
    D = int(np.log2(config.p_size))
    L = starting_length
    H = starting_heigth 

    for s in range(1,D):
            print(s)    
            print(downward_in_channels[s+1])    
            config.downward_dim["in"].update( {s:[L        , H        , downward_in_channels[s  ] ]})  
            config.downward_dim["out"].update({s:[L//(2**s), H//(2**s), downward_in_channels[s+1] ]})
            config.skip_dim.update(           {s:[L//(2**s), H//(2**s), downward_in_channels[s+1] ]})         
            config.upward_dim["out"].update(  {s:[L        , H        , downward_in_channels[s  ] ]})
            config.upward_dim["in"].update(   {s:[    None ,     None ,                     None  ]})
            if s==D-1: 
                config.downward_dim["out"].update({s:[None, None, None]})

    for s in range(1,D):
        if s==D-1: 
            config.upward_dim["in"][s]=[Final_length, Final_length, LDS_Transformer]
        else:
            config.upward_dim["in"][s]=config.upward_dim["out"][s+1]


    # config.downward_in_channels               = {0:None                       , 1:config.input_channel_scale1, 2:config.head_channels, 3:config.head_channels , 4:None}
    # config.upward_in_channels                 = {0:config.input_channel_scale1, 1:config.head_channels       , 2:config.head_channels, 3:config.head_channels , 4:None}
    # config.skip_channels_size                 = {0:config.input_channel_scale1, 1:config.head_channels       , 2:config.head_channels, 3:config.head_channels , 4:None}
    # config.decoder_out_channels               = {0:config.input_channel_scale1, 1:config.input_channel_scale1, 2:config.head_channels, 3:config.head_channels , 4:None}


    # config.downward_dim                       = {'in': {},'out': {}}   
    # config.upward_dim                         = {'in': {},'out': {}}
    # config.skip_dim                           = {}

    #config.n_classes                         = None
    config.activation                         = 'softmax'
    return config

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_output_trans / self.num_attention_heads)
        self.all_head_size       = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_output_trans, self.all_head_size)
        self.key   = Linear(config.hidden_output_trans, self.all_head_size)
        self.value = Linear(config.hidden_output_trans, self.all_head_size)

        self.out          = Linear(config.hidden_output_trans, config.hidden_output_trans)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer= self.query(hidden_states)
        mixed_key_layer  = self.key(hidden_states)
        mixed_value_layer= self.value(hidden_states)

        query_layer      = self.transpose_for_scores(mixed_query_layer)
        key_layer        = self.transpose_for_scores(mixed_key_layer)
        value_layer      = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs  = self.softmax(attention_scores)
        weights          = attention_probs if self.vis else None
        attention_probs  = self.attn_dropout(attention_probs)

        context_layer    = torch.matmul(attention_probs, value_layer)
        context_layer    = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        print("                    new_context_layer_shape ", new_context_layer_shape)
        context_layer    = context_layer.view(*new_context_layer_shape)
        #print("Forward attention : context_layer ", context_layer)
        print("                    context_layer.shape ", context_layer.shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_output_trans, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_output_trans)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size    = config.hidden_output_trans
        self.attention_norm = LayerNorm(config.hidden_output_trans, eps=1e-6)
        self.ffn_norm       = LayerNorm(config.hidden_output_trans, eps=1e-6)
        self.ffn            = Mlp(config)
        self.attn           = Attention(config, vis)
    
    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Embeddings(nn.Module):
    def __init__(self, config, img_size=(32,64), in_channels=672):
        #z.shape ~ something like [32, 672, 32, 64]
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        # img_size    = _pair(img_size)
        img_size    = img_size

        if config.patches.get("grid") is not None:   # ResNet
            grid_size       = config.patches["grid"]
            patch_size      = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches       = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid     = True
        else:
            patch_size  = _pair(config.patches["size"])
            n_patches   = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        # if self.hybrid:
        #     self.hybrid_model    = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
        #     in_channels          = self.hybrid_model.width * 16
        self.patch_embeddings    = Conv2d(in_channels=in_channels,out_channels=config.hidden_size,kernel_size=patch_size,stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout             = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features
    
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size,)
        self.encoder    = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights      = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features

class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config      = config
        head_channels    = config.head_channels
        self.conv_more   = Conv2dReLU(config.hidden_size,head_channels,kernel_size=3,padding=1,use_batchnorm=True,)
        decoder_channels = config.decoder_channels
        in_channels      = [head_channels] + list(decoder_channels[:-1])
        out_channels     = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0
        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=None,
            use_batchnorm=True,
            scale        =None
            ):
        super().__init__()
        
        self.in_ch   = in_channels
        self.out_ch  = out_channels
        self.scale   = scale
        self.skip_ch = 0

        if skip_channels is not None:
            self.skip_ch = skip_channels


        print('in_channels + skip_channels')
        print(in_channels ,  skip_channels)
        self.conv1 = Conv2dReLU(
            self.in_ch + self.skip_ch,
            self.out_ch,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            self.out_ch,
            self.out_ch,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        print('++++++++++++++++++++')
        if self.skip_ch!=0:
            print('Decoderblock x.shape: '   ,x.shape)
            print('Decoderblock skip.shape: ',skip.shape)
            x = torch.cat([x, skip], dim=1)       
        print('******************')
        x = self.conv1(x)        
        x = self.conv2(x)
        return x

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom        = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb      = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias   = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

######################################################################Our version 

class Multi_Scale_Skip_Connection_Block(nn.Module):
    def __init__(self,in_ch,skip_ch,scale):
        super().__init__()
        self.scale        =scale 
        self.in_channels  =in_ch
        self.skip_channels=skip_ch
        self.conv   = nn.Sequential(nn.Conv2d(in_channels =in_ch,
                                              out_channels=skip_ch,kernel_size =3,
                                              padding=1,stride=2,bias=False),
                                    nn.BatchNorm2d(skip_ch),
                                    nn.ReLU())
    def forward(self, x):
        print('Multi_Scale_Skip_Connection_Block self.in_channels'  ,self.in_channels)
        print('Multi_Scale_Skip_Connection_Block self.skip_channels',self.skip_channels)
        print('Multi_Scale_Skip_Connection_Block x.shape',x.shape)
        print('Multi_Scale_Skip_Connection_Block Scale',self.scale)
        x = self.conv(x)
        return x
    
class Multi_Scale_Skip_Connection_Layer(nn.Module):
    def __init__(self, config):
        super(Multi_Scale_Skip_Connection_Layer, self).__init__() 
        in_channels      = config.downward_in_channels                
        skip_channels    = config.skip_channels_size
        blocks           = []
        self.scales      = []
        for scale in config.scales:            
            in_ch  = in_channels[scale]
            skip_ch= skip_channels[scale]
            if (in_ch is not None) and (skip_ch is not None):
                blocks.append(Multi_Scale_Skip_Connection_Block(in_ch, skip_ch,scale))
                self.scales.append(scale)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        # skip=[]
        skip={}
        print('******************')
        print('Multi_Scale_Skip_Connection_Layer X.shape',x.shape)
        skip.update({0:x})
        for i, mssc_block in enumerate(self.blocks):
            scale = mssc_block.scale
            x = mssc_block(x)
            print('Multi_Scale_Skip_Connection_Layer Skip connection block scale',scale)
            print('Multi_Scale_Skip_Connection_Layer Skip connection block skip.shape',x.shape)
            skip.update({scale:x})
        return skip

class Multi_Scale_Skip_Connection_Layer_essai(nn.Module):
    def __init__(self, config):
        super(Multi_Scale_Skip_Connection_Layer_essai, self).__init__() 
        # starting_length        = config.input_size[0]
        # starting_heigth        = config.input_size[1]
        # starting_input_channel = config.input_size[2]
        # downward_in_channels   = config.downward_in_channels
        # Deph                   = config.head_channels
        # LDS_Transformer        = config.hidden_output_trans
        # patch_size             = config.p_size

        # Final_length           = starting_heigth/patch_size
        # Final_heigth           = starting_length/patch_size
        # self.downward_dim          = {'in': {},'out': {}}   
        # self.upward_dim            = {'in': {},'out': {}}
        # self.skip_dim              = {}
        # D = np.log2(config.p_size)
        # L = starting_length
        # H = starting_heigth 

        # for s in range(1,D):        
        #     self.downward_dim["in"].update( {s:[L       , H       , downward_in_channels[s  ] ]})
        #     self.downward_dim["out"].update({s:[L/(2**s), H/(2**s), downward_in_channels[s+1] ]})
        #     self.skip_dim.update(           {s:[L/(2**s), H/(2**s), downward_in_channels[s+1] ]})
        #     self.upward_dim["out"].update(  {s:[L       , H       , downward_in_channels[s  ] ]})
        #     self.upward_dim["in"].update(   {s:[    None,     None,                       None]})
        #     if s==D-1: 
        #         self.downward_dim["out"]={s:[None, None, None]}
        # for s in range(1,D):
        #     if s==D-1: 
        #         self.upward_dim["in"][s]=[Final_length,Final_length,LDS_Transformer]
        #     else:
        #         self.upward_dim["in"][s]=self.upward_dim["out"][s+1]

        # scales={}
        D            = config.n_scales
        downward_dim = config.downward_dim
        skip_dim     = config.skip_dim


        blocks=[]
        for s in range(1,D):
            blocks.append(Multi_Scale_Skip_Connection_Block_essai(downward_dim['in'][s][2],skip_dim[s][2],s))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        skip={}
        print('******************')
        print('Multi_Scale_Skip_Connection_Layer X.shape',x.shape)
        skip.update({0:x})
        for i, mssc_block in enumerate(self.blocks):
            scale = mssc_block.scale
            x = mssc_block(x)
            print('Multi_Scale_Skip_Connection_Layer Skip connection block scale',scale)
            print('Multi_Scale_Skip_Connection_Layer Skip connection block skip.shape',x.shape)
            skip.update({scale:x})
        return skip

class Multi_Scale_Skip_Connection_Block_essai(nn.Module):
    def __init__(self,in_ch,skip_ch,scale):
        super().__init__()
        self.scale        =scale 
        self.in_channels  =in_ch
        self.skip_channels=skip_ch
        self.conv   = nn.Sequential(nn.Conv2d(in_channels =in_ch,
                                              out_channels=skip_ch,kernel_size =3,
                                              padding=1,stride=2,bias=False),
                                    nn.BatchNorm2d(skip_ch),
                                    nn.ReLU())
    def forward(self, x):
        print('Multi_Scale_Skip_Connection_Block self.in_channels'  ,self.in_channels)
        print('Multi_Scale_Skip_Connection_Block self.skip_channels',self.skip_channels)
        print('Multi_Scale_Skip_Connection_Block x.shape',x.shape)
        print('Multi_Scale_Skip_Connection_Block Scale',self.scale)
        x = self.conv(x)
        return x

# class Cnn_Vit_Hybrid_Up_Sampler_essai(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config      = config
#         head_channels    = config.head_channels       
#         self.conv_more   = Conv2dReLU(config.hidden_output_trans,head_channels,kernel_size=3,padding=1,use_batchnorm=True,)

#         # in_channels      = config.upward_in_channels
#         # out_channels     = config.decoder_out_channels 
#         # skip_channels    = config.skip_channels_size

#         in_channels      = config.upward_in_channels
#         out_channels     = config.decoder_out_channels 
#         skip_channels    = config.skip_channels_size

#         blocks     = []
#         self.scale = []
#         for scale in config.scales[::-1]:
#             in_ch   = in_channels[scale]   
#             out_ch  = out_channels[scale]    
#             sk_ch   = skip_channels[scale]   
#             if (in_ch is not None) and (out_ch is not None):
#                 blocks.append(DecoderBlock(in_ch, out_ch, sk_ch))
#                 self.scale.append(scale)

#         self.Blocks = nn.ModuleList(blocks)
#         self.blocks  = {scale:self.Blocks[i] for i,scale in enumerate(self.scale)}
#         print('Cnn_Vit_Hybrid_Up_Sampler  self.blocks:',self.blocks)
    
#     def forward(self, hidden_states, features=None):
#         B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
#         h = self.config.input_size[0]//self.config.patch_size[0]
#         w = self.config.input_size[1]//self.config.patch_size[1]
#         # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
#         x = hidden_states.permute(0, 2, 1)
#         x = x.contiguous().view(B, hidden, h, w)
#         # print('x_shape before conv_more',x.shape)
#         x = self.conv_more(x)
#         # print('x_shape after conv_more',x.shape)     


#         # for i, decoder_block in enumerate(self.blocks):
#         for scale in self.blocks.keys():
#             decoder_block = self.blocks[scale]
#             if features is not None:
#                 skip = features[scale] #if (i < self.config.n_skip) else None
#             else:
#                 skip = None
#             # print(i,'decoder x.shape', x.shape)
#             # if skip is not None:
#             print('decoder decoder_block.scale ', scale)
#             x = decoder_block(x, skip=skip)


#         # for i, decoder_block in enumerate(self.blocks):
#         #     # print("Cnn_Vit_Hybrid_Up_Sampler i decoder:",i)
#         #     print('i, decoder_block', i)
#         #     print('scale', decoder_block.scale)
#         #     if features is not None:
#         #         skip = features[i] if (i < self.config.n_skip) else None
#         #     else:
#         #         skip = None
#         #     # print(i,'decoder x.shape', x.shape)
#         #     # if skip is not None:
#         #     print('decoder skip.scale ', skip.scale)
#         #     x = decoder_block(x, skip=skip)
        
#         return x


class Cnn_Vit_Hybrid_Up_Sampler_essai(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config      = config
        head_channels    = config.head_channels       
        self.conv_more   = Conv2dReLU(config.hidden_output_trans,head_channels,kernel_size=3,padding=1,use_batchnorm=True,)

        # in_channels      = config.upward_in_channels
        # out_channels     = config.decoder_out_channels 
        # skip_channels    = config.skip_channels_size
        in_channels      = {s:v[2] for s,v in config.upward_dim['in'].items()}
        out_channels     = {s:v[2] for s,v in config.upward_dim['out'].items()}
        skip_channels    = {s:v[2] for s,v in config.skip_dim.items()}

        in_channels.update({0:out_channels[1]})
        out_channels.update({0:out_channels[1]})
        skip_channels.update({0:None})

        blocks      = []
        self.scale  = []
        scales_list = [0] + config.scales
        for scale in scales_list[::-1]: #  config.scales[::-1]:
            in_ch   = in_channels[scale] 
            out_ch  = out_channels[scale]  
            sk_ch   = skip_channels[scale] 
            print('scale',scale)
            if (in_ch is not None) and (out_ch is not None):
                print("in_ch is not None) and (out_ch is not None")
                blocks.append(DecoderBlock(in_ch, out_ch, sk_ch))
                self.scale.append(scale)

        self.Blocks = nn.ModuleList(blocks)
        self.blocks  = {scale:self.Blocks[i] for i,scale in enumerate(self.scale)}
        print('Cnn_Vit_Hybrid_Up_Sampler  self.blocks:',self.blocks)
    
    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h = self.config.input_size[0]//self.config.patch_size[0]
        w = self.config.input_size[1]//self.config.patch_size[1]
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        # print('x_shape before conv_more',x.shape)
        x = self.conv_more(x)
        # print('x_shape after conv_more',x.shape)     


        # for i, decoder_block in enumerate(self.blocks):
        for scale in self.blocks.keys():
            decoder_block = self.blocks[scale]
            if features is not None:
                print('feature not none')
                skip = features[scale] #if (i < self.config.n_skip) else None
            else:
                print('NONE !!!')
                skip = None

            # print(i,'decoder x.shape', x.shape)
            # if skip is not None:
            print('decoder decoder_block.scale ', scale)
            x = decoder_block(x, skip=skip)


        # for i, decoder_block in enumerate(self.blocks):
        #     # print("Cnn_Vit_Hybrid_Up_Sampler i decoder:",i)
        #     print('i, decoder_block', i)
        #     print('scale', decoder_block.scale)
        #     if features is not None:
        #         skip = features[i] if (i < self.config.n_skip) else None
        #     else:
        #         skip = None
        #     # print(i,'decoder x.shape', x.shape)
        #     # if skip is not None:
        #     print('decoder skip.scale ', skip.scale)
        #     x = decoder_block(x, skip=skip)
        
        return x


class Cnn_Vit_Hybrid_Embeddings(nn.Module):
    def __init__(self, config, img_size=(32,64), in_channels=672):
        super().__init__()
        self.config     = config
        out_channels    = config.hidden_output_trans
        grid_size       = config.patches["grid"]        
        patch_size      = config.patch_size
        n_patches       = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  

        
        self.patch_embeddings    = Conv2d(in_channels =in_channels,
                                          out_channels=out_channels,
                                          kernel_size =patch_size,
                                          stride      =patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, out_channels))
        self.dropout             = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        features = []
        features.append(x)
        x = self.patch_embeddings(x)                       # (B, hidden, n_patches[0], n_patches[1])
        print('self.patch_embeddings(x)',x.shape)
        x = x.flatten(2)
        print('flatten',x.shape)

        x = x.transpose(-1, -2)                            # (B, n_patches[0], hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        print('embedding.shape',embeddings.shape)
        return embeddings, features


    # def forward(self, x):
    #     if self.hybrid:
    #         x, features = self.hybrid_model(x)
    #     else:
    #         features = None
    #     x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
    #     x = x.flatten(2)
    #     x = x.transpose(-1, -2)  # (B, n_patches, hidden)

    #     embeddings = x + self.position_embeddings
    #     embeddings = self.dropout(embeddings)
    #     return embeddings, features

class Cnn_Vit_Hybrid_Encoder(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.vis          = vis
        self.layer        = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_output_trans, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Cnn_Vit_Hybrid_Up_Sampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config      = config
        head_channels    = config.head_channels       
        self.conv_more   = Conv2dReLU(config.hidden_output_trans,head_channels,kernel_size=3,padding=1,use_batchnorm=True,)

        in_channels      = config.upward_in_channels
        out_channels     = config.decoder_out_channels 
        skip_channels    = config.skip_channels_size    

        blocks     = []
        self.scale = []
        for scale in config.scales[::-1]:
            in_ch   = in_channels[scale] 
            out_ch  = out_channels[scale]  
            sk_ch   = skip_channels[scale] 
            print('scale',scale)
            if (in_ch is not None) and (out_ch is not None):
                print("in_ch is not None) and (out_ch is not None")
                blocks.append(DecoderBlock(in_ch, out_ch, sk_ch))
                self.scale.append(scale)

        self.Blocks = nn.ModuleList(blocks)
        self.blocks  = {scale:self.Blocks[i] for i,scale in enumerate(self.scale)}
        print('Cnn_Vit_Hybrid_Up_Sampler  self.blocks:',self.blocks)
    
    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h = self.config.input_size[0]//self.config.patch_size[0]
        w = self.config.input_size[1]//self.config.patch_size[1]
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        # print('x_shape before conv_more',x.shape)
        x = self.conv_more(x)
        # print('x_shape after conv_more',x.shape)     


        # for i, decoder_block in enumerate(self.blocks):
        for scale in self.blocks.keys():
            decoder_block = self.blocks[scale]
            if features is not None:
                print('feature not none')
                skip = features[scale] #if (i < self.config.n_skip) else None
            else:
                print('NONE')
                skip = None
            # print(i,'decoder x.shape', x.shape)
            # if skip is not None:
            print('decoder decoder_block.scale ', scale)
            x = decoder_block(x, skip=skip)


        # for i, decoder_block in enumerate(self.blocks):
        #     # print("Cnn_Vit_Hybrid_Up_Sampler i decoder:",i)
        #     print('i, decoder_block', i)
        #     print('scale', decoder_block.scale)
        #     if features is not None:
        #         skip = features[i] if (i < self.config.n_skip) else None
        #     else:
        #         skip = None
        #     # print(i,'decoder x.shape', x.shape)
        #     # if skip is not None:
        #     print('decoder skip.scale ', skip.scale)
        #     x = decoder_block(x, skip=skip)
        
        return x

class Cnn_Vit_Hybrid_Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super().__init__()
        self.embeddings = Cnn_Vit_Hybrid_Embeddings(config, img_size=img_size,in_channels=512)
        self.encoder    = Cnn_Vit_Hybrid_Encoder(config,vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights      = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features

class Paradis_Global_Feature_Extraction_Layer(nn.Module):

    def __init__(self, config, img_size=(32,64), vis=True):
        super().__init__()
        print("####################################")
        print("Multi_Scale_Skip_Connection_Layer")
        self.skip_connection = Multi_Scale_Skip_Connection_Layer_essai(config)
        print("Cnn_Vit_Hybrid_Transformer")
        self.transformer     = Cnn_Vit_Hybrid_Transformer(config, img_size, vis)
        print("Cnn_Vit_Hybrid_Up_Sampler")
        self.decoder         = Cnn_Vit_Hybrid_Up_Sampler_essai(config) 
        self.config          = config
        print("####################################")

    def forward(self, x):
        print("####################################")
        print('Paradis_Global_Feature_Extraction_Layer  forward')
        print("####################################")
        x_skip = self.skip_connection(x)
        x_trans, attn_weights, features = self.transformer(x) 
        print("####################################")
        print('Paradis Global x_trans',x_trans.shape)
        print("####################################")
        print('Paradis Global x_skip',[x_s for x_s in x_skip])
        print("####################################")
        x_up = self.decoder(hidden_states=x_trans,features=x_skip)
        print('x_up.shape',x_up.shape)
        return x_up