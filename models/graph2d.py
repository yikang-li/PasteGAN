import torch
import torch.nn as nn
import numpy as np
import math
import glog as log
from .layers import build_mlp
from .layers import build_cnn
from .layers import weights_init
from .layers import get_activation, get_normalization_2d


"""
PyTorch modules for dealing with graphs on 2d feature maps.
"""

class GraphTripleConv2d(nn.Module):
    """
    A single layer of feature map convolution.
    """
    def __init__(self, input_dim, output_dim=128, hidden_dim=None,
                 input_dim_pred=None, output_dim_pred=None,
                 vocab = None,
                 pooling='avg', normalization='instance',
                 activation="leakyrelu-0.1",
                 last_layer=False,
                 use_mask_net=False,
                 use_flow_net=False,
                 valid_edge_only=False,):
        '''
        use_mask_net: apply mask maps to object feature map
        use_flow_net: predicate the per-pixel displacement of the objects
        valid_edge_only: we only use the edges that are not __in_image__
                         to with graph convolution
        '''
        super(GraphTripleConv2d, self).__init__()
        if input_dim_pred is None:
            input_dim_pred = input_dim
        if output_dim_pred is None:
            output_dim_pred = output_dim
        self.input_dim = input_dim
        self.input_dim_pred = input_dim_pred
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_dim_pred = output_dim_pred
        self.pooling = pooling
        self.last_layer = last_layer
        self.valid_edge_only = valid_edge_only
        self.vocab = vocab

        net1_2d_kwargs = {
            'normalization': normalization,
            'padding': 'same',
            'activation': activation,
            'arch': 'I%d,C3-%d-1,C3-%d-1' % (
                2 * self.input_dim + self.input_dim_pred,
                self.hidden_dim,
                self.hidden_dim if self.last_layer
                    else self.hidden_dim * 2 + self.output_dim_pred,
                ),
        }
        self.net1_2d, D = build_cnn(**net1_2d_kwargs)
        self.activation = nn.Sequential(
            get_normalization_2d(self.output_dim_pred, normalization),
            get_activation(activation),
        )

        net2_2d_kwargs = {
            'normalization': normalization,
            'padding': 'same',
            'activation': activation,
            'arch': 'I%d,C3-%d-1,C1-%d-1' % (
                self.hidden_dim,
                self.hidden_dim,
                self.output_dim,
                ),
            'non_linear_activated': True
        }
        self.net2_2d, D = build_cnn(**net2_2d_kwargs)

        net_obj_kwargs = {
            'normalization': normalization,
            'padding': 'same',
            'activation': activation,
            'arch': 'I%d,C3-%d,C1-%d' % (
                self.input_dim,
                self.hidden_dim,
                self.output_dim,
                ),
            'non_linear_activated': True
        }
        self.net_obj, D = build_cnn(**net_obj_kwargs)

        self.use_mask_net = use_mask_net
        if self.use_mask_net:
            mask_net, _ = build_cnn(
                arch="I%d,C1-%d,C3-%d,C1-1" % (
                    self.input_dim,
                    self.input_dim,
                    self.input_dim
                    ),
                normalization=normalization,
                activation=activation,
                non_linear_activated=False,
                padding='same',
            )
            self.mask_net = nn.Sequential(
                mask_net,
                nn.Sigmoid(),
            )

        self.use_flow_net = use_flow_net
        if self.use_flow_net:
            # the flow net predicts the displacement with respective to
            # the original place. Therefore, the output of flownet should
            # add the original coordinates
            flow_net, _ = build_cnn(
                arch="I%d,C3-%d,C3-%d,C3-2" % (
                    self.input_dim,
                    self.input_dim,
                    self.input_dim,
                    ),
                normalization=normalization,
                activation=activation,
                non_linear_activated=False,
                padding='same',
            )
            self.flow_net = nn.Sequential(flow_net, nn.Tanh())
            for m in self.flow_net.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
                    nn.init.normal_(m.weight, mean=0, std=0.01)

        # Attention module
        self.w_attn = 1.0   # attention weights
        attn_1 = {
            'normalization': normalization,
            'padding': 'same',
            'activation': activation,
            'arch': 'I%d,C1-%d-1' % (
                self.hidden_dim,
                self.hidden_dim,
                )
        }
        self.attn_1, D = build_cnn(**attn_1)

        attn_2 = {
            'normalization': normalization,
            'padding': 'same',
            'activation': activation,
            'arch': 'I%d,C1-%d-1' % (
                self.hidden_dim,
                self.hidden_dim,
                )
        }
        self.attn_2, D = build_cnn(**attn_2)

        attn_3 = {
            'normalization': normalization,
            'padding': 'same',
            'activation': activation,
            'arch': 'I%d,C1-%d-1' % (
                self.hidden_dim,
                self.hidden_dim,
                )
        }
        self.attn_3, D = build_cnn(**attn_3)

    def preprocessing_obj(self, obj_maps):
        if self.use_flow_net:
            # transform the feature grid
            transform_grid = self.flow_net(obj_maps)
            transform_grid = torch.transpose(transform_grid, 1, 2)
            transform_grid = torch.transpose(transform_grid, 2, 3)
            # Generate the base grid.
            # Transform grid only predicts the displacement
            theta = torch.zeros(transform_grid.size(0), 2, 3).to(transform_grid.device)
            theta[:, 0, 0] = 1.
            theta[:, 1, 1] = 1.
            base_grid = nn.functional.affine_grid(
                theta=theta, size=obj_maps.size(),
            )
            transform_grid = transform_grid + base_grid
            obj_maps = nn.functional.grid_sample(obj_maps,
                transform_grid, padding_mode="zeros")

        if self.use_mask_net:
            masks = self.mask_net(obj_maps)
            obj_maps = masks * obj_maps
        return obj_maps

    def forward(self, obj_maps, pred_vecs, edges, obj_to_img):
        """
        Inputs:
        - obj_maps: FloatTensor of shape (O, C, h, w) giving map features for all objects
        - pred_vecs: FloatTensor of shape (T, C) giving vectors for all predicates
        - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
          presence of a triple [obj_maps[i], pred_vecs[k], obj_maps[j]]

        Outputs:
        - new_obj_maps: FloatTensor of shape (O, C, h, w) giving new maps for objects
        - new_pred_maps: FloatTensor of shape (O, C, h, w) giving new maps for predicates
        """
        obj_maps = self.preprocessing_obj(obj_maps)

        if self.last_layer:
            return self.forward_last(obj_maps, pred_vecs, edges)

        dtype, device = obj_maps.dtype, obj_maps.device
        O, T = obj_maps.size(0), pred_vecs.size(0)
        H, W = obj_maps.size(2), obj_maps.size(3)
        assert obj_maps.size(1) == self.input_dim
        assert pred_vecs.size(1) == self.input_dim_pred

        if pred_vecs.dim() == 2:
            pred_vecs = pred_vecs.view(T, -1, 1, 1).expand(T, -1, H, W)
        elif pred_vecs.dim() != 4:
            assert False, "pred_vecs in should have 2 or 4 dims."

        # Break apart indices for subjects and objects; these have shape (T,)
        s_idx = edges[:, 0].contiguous()
        o_idx = edges[:, 1].contiguous()

        # Get current features for subjects and objects; these have shape (T, C, h, w)
        cur_s_maps = obj_maps[s_idx]
        cur_o_maps = obj_maps[o_idx]

        # Pass through net1_2d and these have shape (T, C, h, w)
        cur_t_maps = torch.cat([cur_s_maps, pred_vecs, cur_o_maps], dim=1)
        new_t_maps = self.net1_2d(cur_t_maps)

        # the first self.hidden_dim channels are used for subject
        new_s_maps = new_t_maps[:, :self.hidden_dim, :, :]
        # the second self.hidden_dim channels are used for subject
        new_o_maps = new_t_maps[:, self.hidden_dim+self.output_dim:, :, :]
        # the remaining channels are used for subject
        if self.last_layer:
            new_p_maps = None
        else:
            new_p_maps = self.activation(new_t_maps[:, self.hidden_dim:self.hidden_dim+self.output_dim, :, :])

        # Recover the new object maps with shape (O, C, h, w)
        if self.valid_edge_only:
            valid_mask = edges[:, 2] != self.vocab['pred_name_to_idx']['__in_image__']
            s_idx = s_idx[valid_mask]
            o_idx = o_idx[valid_mask]
            new_s_maps = new_s_maps[valid_mask]
            new_o_maps = new_o_maps[valid_mask]

        pooled_obj_maps = torch.zeros(O, self.hidden_dim, H, W,
                            dtype=dtype, device=device)
        s_idx_exp = s_idx.view(-1, 1, 1, 1).expand_as(new_s_maps)
        o_idx_exp = o_idx.view(-1, 1, 1, 1).expand_as(new_o_maps)
        pooled_obj_maps = pooled_obj_maps.scatter_add(0, s_idx_exp, new_s_maps)
        pooled_obj_maps = pooled_obj_maps.scatter_add(0, o_idx_exp, new_o_maps)

        if self.pooling == "avg":
            # Figure out how many times each object has appeared, again using
            # some scatter_add trickery.
            obj_counts = torch.zeros(O, dtype=dtype, device=device)
            ones = torch.ones(T, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, s_idx, ones)
            obj_counts = obj_counts.scatter_add(0, o_idx, ones)

            # Divide the new object vectors by the number of times they
            # appeared, but first clamp at 1 to avoid dividing by zero;
            # objects that appear in no triples will have output vector 0
            # so this will not affect them.
            obj_counts = obj_counts.clamp(min=1)
            pooled_obj_maps = pooled_obj_maps / obj_counts.view(-1, 1, 1, 1)

        # Send pooled object maps through net2 to get output object maps
        obj_maps = self.net2_2d(pooled_obj_maps) + \
                       self.net_obj(obj_maps)

        return obj_maps, new_p_maps

    def forward_last(self, obj_maps, pred_vecs, edges):
        """
        Inputs:
        - obj_maps: FloatTensor of shape (O, C, H, W) giving map features for all objects
        - pred_vecs: FloatTensor of shape (T, C) giving vectors for all predicates
        - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
          presence of a triple [obj_maps[i], pred_vecs[k], obj_maps[j]]

        Outputs:
        - scene_canvas: FloatTensor of shape (N, C, H, W) giving the scene canvas
        """
        dtype, device = obj_maps.dtype, obj_maps.device
        O, T = obj_maps.size(0), pred_vecs.size(0)
        H, W = obj_maps.size(2), obj_maps.size(3)
        assert obj_maps.size(1) == self.input_dim
        assert pred_vecs.size(1) == self.input_dim_pred

        if pred_vecs.dim() == 2:
            pred_vecs = pred_vecs.view(T, -1, 1, 1).expand(T, -1, H, W)
        elif pred_vecs.dim() != 4:
            assert False, "pred_vecs in should have 2 or 4 dims."

        # Break apart indices for subjects and objects; these have shape (T,)
        s_idx = edges[:, 0].contiguous()
        o_idx = edges[:, 1].contiguous()
        # Get current features for subjects and objects; these have shape (T, C, H, W)
        cur_s_maps = obj_maps[s_idx]
        cur_o_maps = obj_maps[o_idx]

        # Pass through net1_2d and these have shape (T, C, H, W)
        new_s_maps = self.attn_1(cur_s_maps)

        # Here we compute the attention
        # we use softmax to get beta along the dimension of channel
        # for each pixel at each channel, the sum is 1.0
        new_pred_vecs = self.attn_2(pred_vecs)
        value = torch.mul(new_s_maps, new_pred_vecs).sum(dim=1, keepdim=True)
        value = value - value.max(0)[0]
        value = value / np.sqrt(new_s_maps.size(1))
        value = value.exp()
        # We use scatter_add as the sum trick
        s_idx_exp = s_idx.view(-1, 1, 1, 1).expand_as(value)
        o_idx_exp = o_idx.view(-1, 1, 1, 1).expand_as(value)
        att_divisor = torch.zeros(O, 1, 1, 1, dtype=dtype, device=device).expand(-1, 1, obj_maps.size(2), obj_maps.size(3))
        att_divisor = att_divisor.scatter_add(0, o_idx_exp, value)
        att_divisor = torch.gather(att_divisor, 0, index=o_idx_exp)
        att_divisor = att_divisor.clamp(min=1e-5)
        beta = value / att_divisor

        # Then, we need to compute the new feature maps using beta
        new_s_maps = torch.mul(beta, self.attn_3(new_s_maps))

        pooled_obj_maps = torch.zeros(O, self.hidden_dim, H, W,
                            dtype=dtype, device=device)
        o_idx_exp = o_idx.view(-1, 1, 1, 1).expand_as(new_s_maps)
        pooled_obj_maps = pooled_obj_maps.scatter_add(0, o_idx_exp, new_s_maps)

        # Figure out how many times each object has appeared, again using
        # some scatter_add trickery.
        obj_counts = torch.zeros(O, dtype=dtype, device=device)
        ones = torch.ones(T, dtype=dtype, device=device)
        obj_counts = obj_counts.scatter_add(0, o_idx, ones)
        image_mask = obj_counts > 0

        scene_canvas = self.w_attn * pooled_obj_maps[image_mask] + obj_maps[image_mask]

        return scene_canvas, None

class GraphTripleConv2dNet(nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, input_dim, output_dim,
                 hidden_dims, vocab, input_dim_pred=None,
                 pooling='avg',
                 normalization='batch',
                 activation='leakyrelu-0.1',
                 transform_residual=False,
                 use_mask_net=False,
                 use_flow_net=False,
                 valid_edge_only=False,):
        '''
        use_mask_net: apply mask (attention) maps to object feature map
        use_flow_net: predicate the per-pixel displacement of the objects
        valid_edge_only: we only use the edges that are not __in_image__
                         to with graph convolution
        '''
        super(GraphTripleConv2dNet, self).__init__()
        if not input_dim_pred:
            input_dim_pred = input_dim
        self.num_layers = len(hidden_dims)
        self.gconvs = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims):
            gconv_kwargs = {
                'input_dim': input_dim,
                'input_dim_pred': input_dim_pred,
                'output_dim': output_dim,
                'hidden_dim': h_dim,
                'pooling': pooling,
                'normalization': normalization,
                'activation': activation,
                'use_mask_net': use_mask_net,
                'use_flow_net': use_flow_net,
                'valid_edge_only': valid_edge_only,
                'vocab': vocab,
            }
            self.gconvs.append(GraphTripleConv2d(**gconv_kwargs))
        self.transform_residual = transform_residual

    def forward(self, obj_maps, pred_vecs, edges, obj_to_img):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            original_obj_maps, original_pred_vecs = obj_maps, pred_vecs
            obj_maps, pred_vecs = gconv(obj_maps, pred_vecs, edges, obj_to_img)
            if self.transform_residual and obj_maps.size(1) == original_obj_maps.size(1):
                obj_maps += original_obj_maps
        return obj_maps, pred_vecs
