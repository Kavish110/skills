import torch
from torch.optim import Optimizer

class Muon(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, muon_norm_clip=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, muon_norm_clip=muon_norm_clip)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                update = exp_avg / denom

                if group['muon_norm_clip'] > 0:
                    update_norm = torch.norm(update)
                    clip_coef = group['muon_norm_clip'] / (update_norm + 1e-6)
                    if clip_coef < 1.0:
                        update.mul_(clip_coef)

                p.add_(update, alpha=-group['lr'])

        return loss

if __name__ == '__main__':
    model = torch.nn.Linear(32, 32).cuda()
    optimizer = Muon(model.parameters(), lr=1e-4)
    x = torch.randn(16, 32, device='cuda')
    y = torch.randn(16, 32, device='cuda')
    for step in range(10):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        print(f'step={step} loss={loss.item():.6f}')
