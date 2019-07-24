import torch
import torch.nn as nn
from .layers import build_mlp

"""
PyTorch modules for dealing with graphs.
"""


def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)


class GraphTripleConv(nn.Module):
    """
    A single layer of scene graph convolution.
    """

    def __init__(self, input_dim, output_dim=None, hidden_dim=512,
                 pooling='avg', mlp_normalization='none',
                 activation='leakyrelu',
                 input_dim_pred=None, output_dim_pred=None,
                 original_gconv=False, vocab=None,
                 category_aware_obj=False,
                 category_aware_rel=False,
                 unary_transform=False,):
        super(GraphTripleConv, self).__init__()
        if input_dim_pred is None:
            input_dim_pred = input_dim
        if output_dim is None:
            output_dim = input_dim
        if output_dim_pred is None:
            output_dim_pred = output_dim
        self.input_dim = input_dim
        self.input_dim_pred = input_dim_pred
        self.output_dim = output_dim
        self.output_dim_pred = output_dim_pred
        self.hidden_dim = hidden_dim

        assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
        self.pooling = pooling
        net1_layers = [2 * input_dim + input_dim_pred,
                       hidden_dim,
                       2 * hidden_dim + output_dim_pred]
        net1_layers = [l for l in net1_layers if l is not None]
        self.net1 = build_mlp(net1_layers,
                              batch_norm=mlp_normalization,
                              activation=activation,
                              final_nonlinearity=original_gconv)
        self.net1.apply(_init_weights)

        net2_layers = [hidden_dim, hidden_dim, output_dim]
        self.net2 = build_mlp(net2_layers,
                              batch_norm=mlp_normalization,
                              activation=activation,
                              start_nonlinearity=not original_gconv)
        self.net2.apply(_init_weights)
        if unary_transform:
            self.object_unary_term = build_mlp(
                [input_dim, output_dim],
                batch_norm = mlp_normalization,
                activation=activation,
                start_nonlinearity=False,
                final_nonlinearity=True,
            )
            self.object_unary_term.apply(_init_weights)
        else:
            self.object_unary_term = None
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        else:
            assert False, "Invalid non-linear activation: {}".format(activation)

    def forward(self, obj_vecs, pred_vecs, edges, objs=None):
        """
        Inputs:
        - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
        - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
        - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
          presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]

        Outputs:
        - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
        - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
        """
        dtype, device = obj_vecs.dtype, obj_vecs.device
        O, T = obj_vecs.size(0), pred_vecs.size(0)
        Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim_pred

        # Break apart indices for subjects and objects; these have shape (T,)
        s_idx = edges[:, 0].contiguous()
        o_idx = edges[:, 1].contiguous()

        # Get current vectors for subjects and objects; these have shape (T, Din)
        cur_s_vecs = obj_vecs[s_idx]
        cur_o_vecs = obj_vecs[o_idx]

        # Get current vectors for triples; shape is (T, 3 * Din)
        # Pass through net1 to get new triple vecs; shape is (T, 2 * H + Dout)
        cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
        new_t_vecs = self.net1(cur_t_vecs)

        # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
        # p vecs have shape (T, Dout)
        new_s_vecs = new_t_vecs[:, :H]
        new_p_vecs = self.activation(new_t_vecs[:, H:(H+Dout)])
        new_o_vecs = new_t_vecs[:, (H+Dout):(2 * H + Dout)]

        # Allocate space for pooled object vectors of shape (O, H)
        pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)

        # Use scatter_add to sum vectors for objects that appear in multiple triples;
        # we first need to expand the indices to have shape (T, D)
        s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
        o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

        if self.pooling == 'avg':
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
            pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

        # Send pooled object vectors through net2 to get output object vectors,
        # of shape (O, self.output_dim)
        new_obj_vecs = self.net2(pooled_obj_vecs)
        if self.object_unary_term is not None:
            new_obj_vecs += self.object_unary_term(obj_vecs)
        return new_obj_vecs, new_p_vecs


class GraphTripleConvNet(nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, input_dim, num_layers=5, hidden_dim=512, pooling='avg',
                 mlp_normalization='none', original_gconv=False,
                 transform_residual=False,
                 vocab=None, unary_transform=False,):
        super(GraphTripleConvNet, self).__init__()
        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        gconv_kwargs = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'pooling': pooling,
            'mlp_normalization': mlp_normalization,
            'original_gconv': original_gconv,
            'vocab': vocab,
            'unary_transform': unary_transform,
        }
        for _ in range(self.num_layers):
            self.gconvs.append(GraphTripleConv(**gconv_kwargs))
        self.transform_residual = transform_residual

    def forward(self, obj_vecs, pred_vecs, edges):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            original_obj_vecs, original_pred_vecs = obj_vecs, pred_vecs
            obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
            if self.transform_residual:
                obj_vecs += original_obj_vecs
                pred_vecs += original_pred_vecs
        return obj_vecs, pred_vecs
