import torch
import torch.nn as nn


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """

    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, a_in, a_out, state_cur):
        a = torch.cat((a_in, a_out, state_cur), 1)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 1)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """

    def __init__(self, opt):
        super(GGNN, self).__init__()

        assert (opt.state_dim >= opt.annotation_dim,
                'state_dim must be no less than annotation_dim')

        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        self.n_edge_types = opt.n_edge_types
        self.n_node = opt.n_node
        self.n_steps = opt.n_steps

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model
        self.propogator = Propogator(
            self.state_dim, self.n_node, self.n_edge_types)

        # Output Model
        self.out = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, 1)
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A):
        for i_step in range(self.n_steps):
            in_states = torch.zeros(prop_state.shape)
            out_states = torch.zeros(prop_state.shape)
            for i in range(self.n_edge_types):
                edge_list = A[i]
                src = edge_list[:, 0]
                dest = edge_list[:, 1]
                src_annotation = torch.index_select(prop_state, 0, src)
                dest_annotation = torch.index_select(prop_state, 0, dest)
                incoming_messages = self.in_fcs[i](src_annotation)
                outgoing_messages = self.out_fcs[i](dest_annotation)
                in_states[dest] = torch.index_select(
                    prop_state, 0, dest) + incoming_messages
                out_states[src] = torch.index_select(
                    prop_state, 0, src) + outgoing_messages
                prop_state = self.propogator(in_states, out_states, prop_state)

        join_state = torch.cat((prop_state, annotation), 1)

        output = self.out(join_state)
        output = output[1:].view(-1, self.n_node, 1)
        output = output.sum(2)

        return output
