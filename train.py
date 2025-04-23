import contextlib
import fire
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import logging
import sys
import os
import torch
from torch import optim, autograd

# Mock the DDP-related functionality for single GPU usage
class SingleGPUWrapper:
    def __init__(self):
        pass
    
    def rank(self):
        return 0
    
    def world_size(self):
        return 1
    
    def reduce_mean(self, tensor):
        return tensor
    
    def wrap_main(self, main_func):
        return main_func

# Create a simplified version of the required libraries
class Libraries:
    def __init__(self):
        self.ddp = SingleGPUWrapper()
        
        # Mock other required libraries
        class Utils:
            def AttributeDict(self, d):
                class AD(dict):
                    def __init__(self, d):
                        super().__init__(d)
                        self.__dict__.update(d)
                    def copy(self):
                        return AD(dict(self))
                return AD(d)
            
            def print_args(self, args):
                logging.info("Arguments:")
                for arg, value in vars(args).items():
                    logging.info(f"  {arg}: {value}")
            
            def train_loop(self, forward, optimizer, steps, names=None, hook=None, 
                         print_freq=10, lr_warmup_steps=0, lr_decay=False, 
                         amp_grad_scaler=False, grad_accum_steps=1, 
                         ddp_models=None, first_step=0, clip_params=None, 
                         clip_quantile=0.95):
                """Simplified training loop for single GPU"""
                step = first_step
                while step < steps:
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Accumulate gradients
                    loss_sum = 0
                    metrics = None
                    for accum_step in range(grad_accum_steps):
                        loss, *metric_values = forward(step=step, accum_step=accum_step, accum_total=grad_accum_steps)
                        loss = loss / grad_accum_steps
                        loss.backward()
                        loss_sum += loss.item()
                        
                        if metrics is None:
                            metrics = [x.item() for x in metric_values]
                        else:
                            metrics = [m + x.item() for m, x in zip(metrics, metric_values)]
                    
                    # Average metrics across accumulation steps
                    metrics = [m / grad_accum_steps for m in metrics]
                    
                    # Optional gradient clipping
                    if clip_params:
                        all_grads = []
                        for param in clip_params:
                            if param.grad is not None:
                                all_grads.append(param.grad.flatten())
                        if all_grads:
                            all_grads = torch.cat(all_grads)
                            clip_value = torch.quantile(all_grads.abs(), clip_quantile)
                            torch.nn.utils.clip_grad_norm_(clip_params, clip_value)
                    
                    # Update weights
                    optimizer.step()
                    
                    # Learning rate schedule
                    if lr_warmup_steps > 0 and step < lr_warmup_steps:
                        # Linear warmup
                        lr_scale = (step + 1) / lr_warmup_steps
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = param_group['original_lr'] * lr_scale
                    elif lr_decay and step >= lr_warmup_steps:
                        # Cosine decay
                        decay_steps = steps - lr_warmup_steps
                        decay_step = step - lr_warmup_steps
                        decay_frac = decay_step / decay_steps
                        lr_scale = 0.5 * (1 + math.cos(math.pi * decay_frac))
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = param_group['original_lr'] * lr_scale
                    
                    # Print metrics
                    if step % print_freq == 0:
                        lr = optimizer.param_groups[0]['lr']
                        log_str = f"Step {step}/{steps}, LR: {lr:.6f}, Loss: {loss_sum:.6f}"
                        if names:
                            for name, value in zip(names, metrics):
                                log_str += f", {name}: {value:.6f}"
                        logging.info(log_str)
                    
                    # Execute hook if provided
                    if hook:
                        hook(step)
                    
                    step += 1
        
        class EMA:
            def __init__(self, module, decay=0.9999):
                self.module = module
                self.decay = decay
                self.shadow_params = {}
                self.collected = False
                self.enabled_context = False
            
            def collect(self):
                if not self.collected:
                    for name, param in self.module.named_parameters():
                        if param.requires_grad:
                            self.shadow_params[name] = param.data.clone()
                    self.collected = True
            
            def step(self):
                if not self.collected:
                    self.collect()
                for name, param in self.module.named_parameters():
                    if param.requires_grad:
                        self.shadow_params[name].lerp_(param.data, 1 - self.decay)
            
            @contextlib.contextmanager
            def enabled(self):
                if not self.collected:
                    self.collect()
                stored_params = {}
                for name, param in self.module.named_parameters():
                    if param.requires_grad:
                        stored_params[name] = param.data.clone()
                        param.data.copy_(self.shadow_params[name])
                try:
                    yield
                finally:
                    for name, param in self.module.named_parameters():
                        if param.requires_grad and name in stored_params:
                            param.data.copy_(stored_params[name])
        
        class DecayToInit:
            def __init__(self, module, strength=0.0):
                self.module = module
                self.strength = strength
                self.init_params = {}
                self.collected = False
            
            def collect(self):
                if not self.collected and self.strength > 0:
                    for name, param in self.module.named_parameters():
                        if param.requires_grad:
                            self.init_params[name] = param.data.clone()
                    self.collected = True
            
            def step(self, current_step, total_steps):
                if self.strength <= 0 or not self.collected:
                    return
                frac = current_step / total_steps
                decay = self.strength * frac
                for name, param in self.module.named_parameters():
                    if param.requires_grad and name in self.init_params:
                        param.data.lerp_(self.init_params[name], decay)
        
        self.utils = Utils()
        self.ema = EMA
        self.decay_to_init = DecayToInit
        
        # Mock models, datasets, and ops modules
        # You'll need to implement these based on your actual code
        class Models:
            def NoiseSchedule(self):
                # Mock implementation
                return torch.nn.Parameter(torch.zeros(10))
            
            def GammaBounds(self, gamma_0, gamma_1):
                # Mock implementation
                return torch.nn.Module()
            
            def EmbeddingMatrix(self, vocab_size, embed_dim):
                # Mock implementation
                return torch.nn.Embedding(vocab_size, embed_dim)
            
            def DiffusionModel(self, dim, embed_dim, n_blocks, n_heads, vocab_size):
                # Mock implementation
                return torch.nn.Module()
        
        class Datasets:
            def get_dataloader(self, dataset, split, batch_size, tokenizer, seq_len, cot=False):
                # Mock implementation
                return []
            
            def get_dataloaders(self, dataset, batch_size, seq_len, cot=False, digit=False, glance=False):
                # Mock implementation
                # Return (train_loader, valid_loader), (word2idx, idx2word), tokenizer
                word2idx = {"<pad>": 0, "<unk>": 1}  # Mock tokenizer vocabs
                idx2word = {0: "<pad>", 1: "<unk>"}
                tokenizer = None  # Mock tokenizer
                return ([], []), (word2idx, idx2word), tokenizer
            
            def infinite_loader(self, loader):
                # Mock implementation
                while True:
                    for batch in loader:
                        yield batch
        
        class Ops:
            def cross_entropy(self, logits, targets):
                # Mock implementation
                return torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    targets.view(-1), 
                    reduction='none'
                ).view(targets.shape)
            
            def gaussian_kl(self, mean1, std1, mean2, std2):
                # Mock implementation
                var1 = std1 ** 2
                var2 = std2 ** 2
                kl = (mean1 - mean2) ** 2 / var2 + var1 / var2 - 1 - torch.log(var1 / var2)
                return 0.5 * kl
        
        self.models = Models()
        self.datasets = Datasets()
        self.ops = Ops()

# Create a global instance of our libraries
lib = Libraries()

# Mock for MuP
class MuP:
    def set_base_shapes(self, main_module, base_module, delta=None):
        # Mock implementation
        pass
    
    class MuAdam:
        def __init__(self, params, impl=None, **kwargs):
            self.optimizer = impl([{'params': group['params'], 
                                   'lr': group.get('lr', 1e-3),
                                   'weight_decay': group.get('weight_decay', 0),
                                   'original_lr': group.get('lr', 1e-3)} 
                                  for group in params], **kwargs)
            
        def zero_grad(self):
            self.optimizer.zero_grad()
            
        def step(self):
            self.optimizer.step()
            
        @property
        def param_groups(self):
            return self.optimizer.param_groups

mup = MuP()

# Mock for evaluation functions
def evaluate(infer_args, test_loader, tokenizer, modules, log_interval=False):
    # Mock implementation
    logging.info("Running mock evaluation...")
    return 0.75  # Mock accuracy

def generate_samples(x, src_mask, modules, infer_args, timesteps_togo):
    # Mock implementation
    return x  # Just return the input as is

# Main training code (modified for single GPU)
def masked_loss(loss, mask, weight, dim=None):
    loss = loss.masked_fill(~mask, 0)
    loss = loss * weight
    average_loss = loss.sum(dim)/(mask.sum(dim)+0.01)
    return average_loss

def set_args(args):
    if not hasattr(args, 'grad_accum_steps'):
        args.grad_accum_steps = 1
    if not hasattr(args, 'cot'):
        args.cot = False
    if not hasattr(args, 'digit'):
        args.digit = True

    bs = args.batch_size*args.grad_accum_steps
    save_weights_path=f"outputs/{args.dataset}-bs{bs}"
    if args.fix_src:
        save_weights_path += '-fix_src'
    if args.cot:
        save_weights_path += '-cot'
    if args.digit:
        save_weights_path += '-digit'
    args.save_weights_path = save_weights_path + f'-steps{args.steps}'

    os.makedirs(args.save_weights_path, exist_ok=True)
    args.train_log = os.path.join(args.save_weights_path, "train.log")
    if os.path.exists(args.train_log): 
        os.remove(args.train_log)

    targets = logging.StreamHandler(sys.stdout), logging.FileHandler(args.train_log, mode='w')
    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO, handlers=targets)


def sampling_gold_prob(i, steps, min_prob=0.1):
    return (1-min_prob)*(steps-i)/steps + min_prob


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(**args):
    args = lib.utils.AttributeDict(args)
    args.setdefault('batch_size', 16)
    args.setdefault('dataset', 'gsm8k')
    args.setdefault('grad_accum_steps', 1)
    args.setdefault('hook_freq', 500)
    args.setdefault('lr', 3e-4)
    args.setdefault('lr_warmup_steps', 25)
    args.setdefault('bias_warmup_steps', 50)
    args.setdefault('lr_decay', True)
    args.setdefault('print_freq', 10)
    args.setdefault('save_weights', True)
    args.setdefault('steps', 9000)
    args.setdefault('weights_path', None)
    args.setdefault('reconst_weight', 1.0)
    args.setdefault('dim', 2048)
    args.setdefault('n_blocks', 24)
    args.setdefault('n_heads', 32)
    args.setdefault('gamma_0', -3.)
    args.setdefault('gamma_1', 6.)
    args.setdefault('embed_dim', 16)
    args.setdefault('seq_len', 256)
    args.setdefault('weight_decay', 4e-5)
    args.setdefault('first_step', 0)
    args.setdefault('auto_resume', False)
    args.setdefault('decay_to_init', 0.)
    args.setdefault('ema', 0.)
    args.setdefault('beta1', 0.9)
    args.setdefault('beta2', 0.99)
    args.setdefault('selfcond', True)
    args.setdefault('clip_quantile', 0.95)
    args.setdefault('reconst_bs_ema', 0.997)
    args.setdefault('fix_src', False)
    args.setdefault('cot', False)
    args.setdefault('digit', True)
    args.setdefault('min_prob', 1.)
    args.setdefault('glance', False)

    set_args(args)
    lib.utils.print_args(args)

    set_seed(2024)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set default dtype to float64
    torch.set_default_dtype(torch.float64)

    if not hasattr(args, 'seq_len'):
        args.seq_len = 256
    if not hasattr(args, 'glance'):
        args.glance = False

    # Print a notice that we're using mock data loaders
    logging.info("NOTICE: Using mock data loaders. Replace these with your actual data loaders.")
    (train_loader, valid_loader), (word2idx, idx2word), tokenizer = lib.datasets.get_dataloaders(
        args.dataset, args.batch_size, args.seq_len, args.cot, args.digit, args.glance
    )
    train_iterator = lib.datasets.infinite_loader(train_loader)

    test_loader = lib.datasets.get_dataloader(args.dataset, 'test', 84, tokenizer, args.seq_len, args.cot)

    logging.info(f"world size: {lib.ddp.world_size()}")
    
    # Mock vocab size - replace with actual vocab size from your dataset
    vocab_size = len(word2idx)
    logging.info(f'vocab_size: {vocab_size}')

    def create_modules(dim, n_heads):
        return {
            'noise_schedule': lib.models.NoiseSchedule().float(),
            'gamma_bounds': lib.models.GammaBounds(args.gamma_0, args.gamma_1).float(),
            'embedding_matrix': lib.models.EmbeddingMatrix(vocab_size, args.embed_dim).float(),
            'model': lib.models.DiffusionModel(dim, args.embed_dim, args.n_blocks, n_heads, vocab_size).float()
        }
    
    if not hasattr(args, 'dim'):
        args.dim = 2048
    if not hasattr(args, 'n_heads'):
        args.n_heads = 32
    modules = create_modules(args.dim, args.n_heads)
    base_modules = create_modules(256, 4)
    delta_modules = create_modules(128, 2)
    for key in modules:
        main, base, delta = modules[key], base_modules[key], delta_modules[key]
        mup.set_base_shapes(main, base, delta=delta)
        main.cuda()
        logging.info(key+':')
        logging.info(f"Module initialized (mock implementation)")

    def load_weights(weights_path):
        logging.info(f'Loading weights from {weights_path}')
        try:
            for name, module in modules.items():
                module.load_state_dict(torch.load(
                    os.path.join(weights_path, f'{name}.pt'),
                    map_location=torch.device('cuda')
                ))
            logging.info("Successfully loaded weights")
        except Exception as e:
            logging.error(f"Error loading weights: {e}")
            logging.info("Continuing with randomly initialized weights")

    first_step = args.first_step
    if args.auto_resume and os.path.exists('model.pt'):
            load_weights('.')
            with open('step', 'r') as f:
                first_step = int(f.read()) + 1
    elif args.weights_path is not None:
        load_weights(args.weights_path)

    logging.info(f'Starting from step {first_step}')

    # For single GPU, we don't need DDP
    ddp_modules = {name: module for name, module in modules.items()}

    logging.info('Running in single GPU mode')

    emas = {
        name: lib.ema.EMA(module, args.ema)
        for name, module in modules.items()
    }

    decay_to_init = {
        name: lib.decay_to_init.DecayToInit(module, args.decay_to_init)
        for name, module in modules.items()
    }

    loss_ema_bias     = torch.tensor(1e-8).cuda()
    reconst_ema       = torch.tensor(1e-8).cuda()
    diffusion_ema     = torch.tensor(1e-8).cuda()
    reconst_sqr_ema   = torch.tensor(1e-8).cuda()
    diffusion_sqr_ema = torch.tensor(1e-8).cuda()
    reconst_bs_cache  = {}

    # infer_args for scheduled sampling and evaluation
    infer_args = args.copy()
    infer_args.update({'initial_noise_scale': 1.0,
                    'sampling_timesteps': 16, 
                    'score_temp': 0.5,
                    'dpm_solver': False,
                    'logit_sample': False,
                    'logit_temp': 0.5,
                    'runs': 1,
                    'apply_sc': False,
                    'cot_steps': 6,
                    'limit': False
    })
    infer_args = lib.utils.AttributeDict(infer_args)

    def forward(step=None, accum_step=None, accum_total=None, x_eval=None):
        """
        Train mode: step, accum_step (0~8), accum_total (8 gpus*1 grad_acc_steps)
        Eval mode: x_eval
        """
        nonlocal reconst_ema, diffusion_ema, reconst_sqr_ema, diffusion_sqr_ema

        train_mode = (x_eval is None)
        if train_mode:
            try:
                x, attn_mask, src_mask = next(train_iterator)
                x = x.cuda()
                attn_mask = attn_mask.cuda()
                src_mask = src_mask.cuda()
            except Exception as e:
                # Create mock data for development
                logging.debug(f"Using mock data due to: {e}")
                batch_size = args.batch_size
                seq_len = args.seq_len
                x = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
                attn_mask = torch.ones((batch_size, seq_len), dtype=torch.bool).cuda()
                src_mask = torch.zeros((batch_size, seq_len, 1), dtype=torch.bool).cuda()
                src_mask[:, :10, :] = True  # First 10 tokens are source
           
            batch_size = x.shape[0] * accum_total
            if step not in reconst_bs_cache:
                # For single GPU, we don't need to synchronize EMA vars
                reconst_bs = int(batch_size / 8)  # 1/8 of batch for reconstruction loss
                reconst_bs = max(1, reconst_bs)
                reconst_bs_cache[step] = reconst_bs
            reconst_bs = reconst_bs_cache[step]
            avg_reconst_bs = float(reconst_bs)
        else:
            try:
                x, attn_mask, src_mask = x_eval
                x = x.cuda()
                attn_mask = attn_mask.cuda()
                src_mask = src_mask.cuda()
            except Exception as e:
                # Create mock data for evaluation
                logging.debug(f"Using mock eval data due to: {e}")
                batch_size = 8
                seq_len = args.seq_len
                x = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
                attn_mask = torch.ones((batch_size, seq_len), dtype=torch.bool).cuda()
                src_mask = torch.zeros((batch_size, seq_len, 1), dtype=torch.bool).cuda()
                src_mask[:, :10, :] = True  # First 10 tokens are source
                
            batch_size = x.shape[0]
            reconst_bs = (batch_size // 8)  # no reconst loss if bs <=8
            reconst_bs += int(np.random.binomial(1, (batch_size % 8) / 8.))
            avg_reconst_bs = batch_size / 8.

        # Generate mock forward pass results
        # In a real implementation, you would call your actual model here
        
        # Mock data for demonstration
        loss = torch.tensor(0.5, requires_grad=True).cuda()
        nll = torch.tensor(0.8).cuda()
        reconst_loss = torch.ones(reconst_bs).cuda()
        prior_loss = torch.tensor(0.3).cuda()
        gamma_0 = torch.tensor(args.gamma_0).cuda()
        gamma_1 = torch.tensor(args.gamma_1).cuda()
        
        return (
            loss,
            nll,
            reconst_loss.sum() / avg_reconst_bs,
            prior_loss,
            gamma_0,
            gamma_1,
            torch.tensor(reconst_bs).cuda(),
        )

    learning_rates = {
        'model': args.lr,
        'noise_schedule': 1e-2,
        'gamma_bounds': 1e-2,
        'embedding_matrix': 1e-2,
    }

    weight_decays = {
        'model': args.weight_decay,
        'noise_schedule': 0.,
        'gamma_bounds': 1e-3,
        'embedding_matrix': 0.,
    }

    def optimizer_impl(param_groups, **kwargs):
        assert('weight_decay' not in kwargs)
        modules_seen = set()
        for i, param_group in enumerate(param_groups):
            weight_decay_set = False
            for name in modules:
                group_params = param_group['params']
                module_params = list(modules[name].parameters())
                if all([any([p is p2 for p2 in module_params]) for p in group_params]):
                    assert(not weight_decay_set)
                    assert(param_group['weight_decay'] == 0.)
                    param_group['weight_decay'] = (
                        weight_decays[name] / (param_group['lr']+1e-16)
                    )
                    weight_decay_set = True
                    modules_seen.add(name)
            assert(weight_decay_set)
        assert(all([name in modules_seen for name in modules]))

        # Use regular AdamW instead of distributed optimizer
        return optim.AdamW(param_groups, **kwargs)

    param_groups = [
        {'params': modules[name].parameters(), 'lr': learning_rates[name], 'weight_decay': 0.}
        for name in modules
    ]
    opt = mup.MuAdam(param_groups, impl=optimizer_impl, betas=(args.beta1, args.beta2))

    def compute_nll(data_iterator, seq_len=args.seq_len):
        with contextlib.ExitStack() as stack:
            for ema in emas.values():
                stack.enter_context(ema.enabled())
            stack.enter_context(torch.no_grad())
            total_nll = 0.
            n = 0
            for i, X in enumerate(data_iterator):
                nll = forward(x_eval=X)[1]
                total_nll += nll.item()
                n += 1
                if i == 10:  # Limited to 10 batches for mock test
                    break
            if n == 0:
                # Mock data if iterator is empty
                return 2.5
        return total_nll/n

    all_val_nlls = []
    all_test_accs = []
    def hook(step):
        for decay in decay_to_init.values():
            decay.step(step, args.steps)

        for ema in emas.values():
            ema.step()

        if step % args.hook_freq == (args.hook_freq - 1):
            try:
                val_nll = compute_nll(iter(valid_loader))
            except Exception as e:
                logging.info(f"Error computing validation NLL: {e}")
                val_nll = 2.5  # Mock value
                
            logging.info(f'NLL (val, seq_len={args.seq_len}): {val_nll}')
            all_val_nlls.append(val_nll)

            # Mock evaluation on test set
            acc = evaluate(infer_args, test_loader, tokenizer, modules, log_interval=False)
            all_test_accs.append(acc)
            logging.info(f'Test accuracy: {acc}')

            # Save weights
            if args.save_weights:
                for name in modules:
                    with emas[name].enabled():
                        torch.save(modules[name].state_dict(), f'{args.save_weights_path}/{name}.pt')
                with open(f'{args.save_weights_path}/step', 'w') as f:
                    f.write(str(step))
                logging.info('Saved weights!')

                plt.clf()
                plt.plot(all_test_accs)
                plt.savefig(f'{args.save_weights_path}/test_acc.jpg')

                plt.clf()
                plt.plot(all_val_nlls)
                plt.savefig(f'{args.save_weights_path}/val_nll.jpg')
                
    logging.info('Starting train loop...')
    lib.utils.train_loop(
        forward,
        opt,
        args.steps,
        names=['nll','reconst','prior','gamma_0','gamma_1','reconst_bs'],
        hook=hook,
        print_freq=args.print_freq,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay=args.lr_decay,
        amp_grad_scaler=False,
        grad_accum_steps=args.grad_accum_steps,
        ddp_models=ddp_modules.values(),
        first_step=first_step,
        clip_params=[
            param
            for module in modules.values()
            for param in module.parameters()
        ],
        clip_quantile=args.clip_quantile,
    )

    final_val_nll = compute_nll(iter(valid_loader))
    logging.info(f'Final val NLL: {final_val_nll}')

    ## evaluate on test set
    test_args = infer_args.copy()
    test_args.update({'sampling_timesteps': 64, 'cot_steps': 12})
    test_args = lib.utils.AttributeDict(test_args)
    
    final_test_acc = evaluate(test_args, test_loader, tokenizer, modules, log_interval=False)
    logging.info(f'Final test accuracy: {final_test_acc}')

    return all_val_nlls, final_val_nll

if __name__ == '__main__':
    fire.Fire(main)