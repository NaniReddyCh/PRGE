import torch
from torch.optim.optimizer import Optimizer

class PRGEOptimizer(Optimizer):
    def __init__(self, params, lr=1e-4, q=4, epsilon=1e-2):
        defaults = dict(lr=lr, q=q, epsilon=epsilon)
        super().__init__(params, defaults)
        self.q = q
        self.epsilon = epsilon

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure required for PRGEOptimizer.step()")

        eval_results = closure()
        losses, seeds = eval_results['losses'], eval_results['seeds']
        projected_grads = (losses[:,0] - losses[:,1]) / (2*self.epsilon)

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if not p.requires_grad: continue
                total_update = torch.zeros_like(p.data)
                for i in range(self.q):
                    torch.manual_seed(seeds[i])
                    noise = torch.randn_like(p.data)
                    total_update += projected_grads[i] * noise
                p.data.add_(total_update, alpha=-lr/self.q)
