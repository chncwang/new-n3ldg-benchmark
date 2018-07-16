class ChildSumTreeLSTM(nn.Module):
    def __init__(self, hyperparameter):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = hyperparameter.embed_dim
        self.hidden_dim = hyperparameter.hidden_dim
        self.out_dim = hyperparameter.n_label
        self.add_cuda = hyperparameter.cuda
        self.hyperparameter = hyperparameter

        self.ix = nn.Linear(self.in_dim, self.hidden_dim)
        self.ih = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.fx = nn.Linear(self.in_dim, self.hidden_dim)
        self.fh = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.ox = nn.Linear(self.in_dim, self.hidden_dim)
        self.oh = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.ux = nn.Linear(self.in_dim, self.hidden_dim)
        self.uh = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.out = nn.Linear(self.hidden_dim, self.out_dim)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss_func = nn.NLLLoss()

        if hyperparameter.cuda:
            self.loss_func.cuda()

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = F.torch.sum(torch.squeeze(child_h, 1), 0)

        i = F.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs) + self.uh(child_h_sum))

        # add extra singleton dimension
        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = F.sigmoid(f)

        # f = F.torch.unsqueeze(f,1) # comment to fix dimension missmatch
        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)

        c = F.torch.mul(i, u) + F.torch.sum(fc, 0)
        h = F.torch.mul(o, F.tanh(c))

        return c, h

    def forward(self, tree, embeds):
        # embeds = self.embedding(embeds)
        if self.add_cuda:
            loss = autograd.Variable(torch.zeros(1)).cuda()
        else:
            loss = autograd.Variable(torch.zeros(1))
        for child in tree.children:
            _, child_loss = self.forward(child, embeds)
            loss = loss + child_loss
        child_c, child_h = self.get_child_states(tree)
        tree.state = self.node_forward(embeds[tree.value-1], child_c, child_h)

        output = self.out(tree.state[1])
        output = self.softmax(output)
        if tree.label is not None:
            # print(map_label_to_target_sentiment(tree.gold_label,num_classes=2,fine_grain=0))
            if self.add_cuda:
                gold = autograd.Variable(torch.LongTensor([tree.label])).cuda()
            else:
                gold = autograd.Variable(torch.LongTensor([tree.label]))
            loss = loss + self.loss_func(output, gold)

        return output, loss

    def get_child_states(self, tree):
        """
        get c and h of all children
        :param tree:
        :return:
        """
        num_children = len(tree.children)
        if num_children == 0:
            child_c = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
            child_h = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
            if self.add_cuda:
                child_c, child_h = child_c.cuda(), child_h.cuda()
        else:
            child_c = autograd.Variable(torch.zeros(num_children, 1, self.hidden_dim))
            child_h = autograd.Variable(torch.zeros(num_children, 1, self.hidden_dim))
            for idx, child in enumerate(tree.children):
                child_c[idx] = child.state[0]
                child_h[idx] = child.state[1]
            if self.add_cuda:
                child_c, child_h = child_c.cuda(), child_h.cuda()
        return child_c, child_h