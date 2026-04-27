import os
import shutil
import subprocess
import sys
from privacy_tools import get_sigma
import argparse

parser = argparse.ArgumentParser(description='Fine-tuning BERT with differentially private compactor')

parser.add_argument('--task', default='SST-2', type=str , choices=['MNLI', 'QNLI', 'QQP', 'SST-2', 'SNLI'], help='name of the downstream task')
parser.add_argument('--gpu_id', default=0, type=int, help='which GPU to use, current implementation only supports using a single GPU')
parser.add_argument('--to_console', action='store_true', help='output to console, for debug use')
parser.add_argument('--sess', type=str, default='default', help='session name')
parser.add_argument('--save_root', type=str, default='log_dir', help='root directory for logs and checkpoints')

#normal hyperparameters
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--batch_size', default=2000, type=int, help='batch size')
parser.add_argument('--epoch', default=50, type=int, help='number of epochs')
parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
parser.add_argument('--arch', default='roberta.base', type=str, choices=['roberta.base', 'roberta.large'], help='model architecture')
# 0.999 is the value used in the original BERT paper
parser.add_argument('--adam_beta2', default=0.999, type=float, help='second beta value of adam')

parser.add_argument('--max_sentences', default=50, type=int, help='max sentences per step. Use a smaller value if your GPU runs out of memory')
parser.add_argument('--max_tokens', default=8000, type=int, help='max tokens per step. Use a smaller value if your GPU runs out of memory')
parser.add_argument('--num_workers', default=4, type=int, help='number of dataloader workers')
parser.add_argument('--validate_interval_updates', default=20, type=int, help='run validation every N updates')

#new hyperparameters
parser.add_argument('--eps', default=8, type=float, help='DP parameter epsilon')
parser.add_argument('--delta', default=1e-5, type=float, help='DP parameter delta')
parser.add_argument('--clip', default=10., type=float, help='clipping threshold of individual gradients')
parser.add_argument('--k', default=16, type=int, help='number of bottleneck dimension')
parser.add_argument('--accountant', default='prv', type=str, choices=['moments', 'prv'], help='privacy accounting method')
parser.add_argument('--active_layers', default='all', type=str, help='active LoRA layers, e.g. "0,1,2,5" or "all"')
parser.add_argument('--route_method', default='uniform', type=str, help='label for layer routing strategy')
parser.add_argument('--dry_run', default=False, action='store_true', help='print trainable param summary and exit before training')
parser.add_argument('--lora_mode', default='standard', type=str, choices=['standard', 'shared_right'], help='LoRA mode')
parser.add_argument('--shared_modules', default='attn', type=str, help='shared modules: attn or attn,ffn')
parser.add_argument('--lora_modules', default='attn,ffn', type=str, help='LoRA modules to train: attn / ffn / attn,ffn')
parser.add_argument('--max_update', default=0, type=int, help='max update steps (0 means disabled)')
parser.add_argument('--debug_param_list', action='store_true', help='print full optimizer parameter list')
parser.add_argument('--debug_grad_norm_once', action='store_true', help='print key grad norms after first backward')
parser.add_argument('--debug_lora_forward_once', action='store_true', help='print LoRA forward branch norms once')
parser.add_argument('--debug_lora_init_check', action='store_true', help='print LoRA init info and selection in dry-run')
parser.add_argument('--debug_param_update_once', action='store_true', help='print key parameter delta after one optimizer step')
parser.add_argument('--dp_hook_only', action='store_true', help='enable batch_grad path but disable clipping/noise')
parser.add_argument('--no_batch_grad_hook', action='store_true', help='disable batch_grad hooks in LoRA and cls head')

# 新增：无DP开关
parser.add_argument('--no_dp', default=False, action='store_true', help='disable differential privacy')
parser.add_argument('--use_canary', default=False, action='store_true', help='use canary-augmented dataset')

parser.add_argument('--fp32', action='store_true', help='use full precision or not')
parser.add_argument('--min_free_gb', default=2.0, type=float, help='minimum free disk space required for checkpoint directory')
parser.add_argument('--full_checkpoint', action='store_true', help='save all checkpoint artifacts')

args = parser.parse_args()

assert args.task in ['MNLI', 'QNLI', 'QQP', 'SST-2', 'SNLI']

if args.use_canary:
    candidates = ['../glue_data/%s-canary-bin'%args.task, './glue_data/%s-canary-bin'%args.task]
else:
    candidates = ['../glue_data/%s-bin'%args.task, './glue_data/%s-bin'%args.task]
data_dir = candidates[0]
for candidate in candidates:
    if os.path.exists(os.path.join(candidate, 'input0', 'dict.txt')):
        data_dir = candidate
        break
output_dir = args.save_root
ckpt_dir = '../%s/model.pt'%args.arch

assert args.batch_size % args.max_sentences == 0
update_freq = args.batch_size // args.max_sentences

dataset_size_dict ={'MNLI':392702, 'QQP':363849, 'QNLI':104743, 'SST-2':67750, 'SNLI':549367}
dataset_size = dataset_size_dict[args.task]

# 修改：sigma计算逻辑
if args.no_dp:
    sigma = -1
    eps = -1
    print('Running without differential privacy')
elif args.eps > 0:
    q = args.batch_size/dataset_size
    steps = args.epoch * (dataset_size//args.batch_size)
    sigma, eps = get_sigma(q, steps, args.eps, args.delta, mode=args.accountant)
    if args.accountant == 'moments':
        from prv_accountant import Accountant
        accountant = Accountant(
            noise_multiplier=sigma,
            sampling_probability=q,
            delta=args.delta,
            eps_error=0.1,
            max_compositions=steps)       
        eps_low, eps_estimate, eps_upper = accountant.compute_epsilon(num_compositions=steps)
        prv_eps = eps_upper
    
    print('Noise standard deviation:', sigma, 'Epsilon: ', eps, 'Delta: ', args.delta)
    if args.accountant == 'moments':
        print('Epsilon will be %.3f if using PRV accountant.'%prv_eps)
else:
    sigma = -1
    eps = -1

sess = args.sess

task_output_dir = os.path.join(output_dir, args.task)
save_dir = os.path.join(task_output_dir, sess)
log_file = os.path.join(task_output_dir, '%s_train_log.txt' % sess)
lock_file = os.path.join(task_output_dir, f'{sess}.lock')
disk_check_dir = task_output_dir if os.path.exists(task_output_dir) else output_dir
if disk_check_dir == '':
    disk_check_dir = '.'
free_bytes = shutil.disk_usage(disk_check_dir).free
min_free_bytes = int(args.min_free_gb * (1024 ** 3))
if free_bytes < min_free_bytes:
    raise OSError('[Errno 28] No space left on device: free {:.2f} GB < required {:.2f} GB ({})'.format(
        free_bytes / (1024 ** 3), args.min_free_gb, os.path.abspath(disk_check_dir)))
os.makedirs(save_dir, exist_ok=True)


def _pid_is_alive(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_sess_lock(path):
    """
    Ensure one active run_exp/train pipeline per session.
    Returns True if lock acquired by current process, False if another process holds it.
    """
    current_pid = os.getpid()
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(str(current_pid))
            return True
        except FileExistsError:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    owner_pid = int((f.read() or "0").strip() or "0")
            except Exception:
                owner_pid = 0
            if _pid_is_alive(owner_pid):
                print(f"SKIP: session lock is held by pid={owner_pid}, sess={sess}")
                return False
            try:
                os.remove(path)
            except OSError:
                print(f"SKIP: lock exists and cannot be removed, sess={sess}")
                return False


def _release_sess_lock(path):
    current_pid = os.getpid()
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            owner_pid = int((f.read() or "0").strip() or "0")
    except Exception:
        owner_pid = 0
    if owner_pid == current_pid and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass

apdx = ' '
# new added hyparameters
apdx += ' --sigma %f --clip %f --k %d '%(sigma, args.clip, args.k)
apdx += ' --active-layers %s --route-method %s ' % (args.active_layers, args.route_method)
apdx += ' --lora-mode %s --shared-modules %s ' % (args.lora_mode, args.shared_modules)
apdx += ' --lora-modules %s ' % args.lora_modules
if args.dry_run:
    apdx += ' --dry-run '
metric='accuracy'
n_classes=2
if args.task == 'MNLI':
    n_classes = 3
    apdx += ' --valid-subset valid,valid1 '
elif args.task == 'SNLI':
    n_classes = 3

if 'base' in args.arch:
    args.arch = 'roberta_base'
else:
    args.arch = 'roberta_large'

if not args.fp32:
    apdx += ' --fp16  --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 '
if not args.full_checkpoint:
    apdx += ' --no-epoch-checkpoints --no-save-optimizer-state --keep-last-epochs 1 '

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
if args.no_batch_grad_hook:
    os.environ['DISABLE_BATCH_GRAD_HOOK'] = '1'
else:
    os.environ.pop('DISABLE_BATCH_GRAD_HOOK', None)
train_cmd = [
    "python", "train.py", data_dir,
    "--save-dir", save_dir,
    "--restore-file", ckpt_dir,
    "--max-positions", "512",
    "--update-freq", str(update_freq),
    "--max-sentences", str(args.max_sentences),
    "--max-tokens", str(args.max_tokens),
    "--task", "sentence_prediction",
    "--num-workers", str(args.num_workers),
    "--distributed-world-size", "1",
    "--distributed-no-spawn",
    "--reset-optimizer", "--reset-dataloader", "--reset-meters",
    "--required-batch-size-multiple", "1",
    "--init-token", "0", "--separator-token", "2",
    "--arch", args.arch,
    "--criterion", "sentence_prediction",
    "--num-classes", str(n_classes),
    "--dropout", "0.1", "--attention-dropout", "0.1",
    "--weight-decay", str(args.weight_decay),
    "--optimizer", "adam", "--adam-betas", f"(0.9,{args.adam_beta2})", "--adam-eps", "1e-06",
    "--clip-norm", "0", "--validate-interval-updates", str(args.validate_interval_updates),
    "--lr-scheduler", "polynomial_decay", "--lr", str(args.lr), "--warmup-ratio", "0.06", "--sess", args.sess,
    "--max-epoch", str(args.epoch), "--seed", str(args.seed), "--no-progress-bar", "--log-interval", "100",
    "--find-unused-parameters", "--skip-invalid-size-inputs-valid-test", "--truncate-sequence", "--embedding-normalize",
    "--tensorboard-logdir", ".", "--bert-pooler", "--pooler-dropout", "0.1",
    "--best-checkpoint-metric", metric, "--maximize-best-checkpoint-metric",
    "--sigma", str(sigma), "--clip", str(args.clip), "--k", str(args.k),
    "--active-layers", args.active_layers, "--route-method", args.route_method,
    "--lora-mode", args.lora_mode, "--shared-modules", args.shared_modules, "--lora-modules", args.lora_modules,
]
if args.max_update > 0:
    train_cmd.extend(["--max-update", str(args.max_update)])
if args.debug_param_list:
    train_cmd.append("--debug-param-list")
if args.debug_grad_norm_once:
    train_cmd.append("--debug-grad-norm-once")
if args.debug_lora_forward_once:
    train_cmd.append("--debug-lora-forward-once")
if args.debug_lora_init_check:
    train_cmd.append("--debug-lora-init-check")
if args.debug_param_update_once:
    train_cmd.append("--debug-param-update-once")
if args.dp_hook_only:
    train_cmd.append("--dp-hook-only")
if args.no_batch_grad_hook:
    train_cmd.append("--no-batch-grad-hook")
if os.name == 'nt':
    train_cmd.extend(["--distributed-backend", "gloo"])
if args.dry_run:
    train_cmd.append("--dry-run")
if not args.fp32:
    train_cmd.extend(["--fp16", "--fp16-init-scale", "4", "--threshold-loss-scale", "1", "--fp16-scale-window", "128"])
if not args.full_checkpoint:
    train_cmd.extend(["--no-epoch-checkpoints", "--no-save-optimizer-state", "--keep-last-epochs", "1"])
if args.task == 'MNLI':
    train_cmd.extend(["--valid-subset", "valid,valid1"])

print('train cmd:', " ".join(train_cmd))
if not _acquire_sess_lock(lock_file):
    sys.exit(0)
try:
    with open(log_file, "a", encoding="utf-8") as lf:
        if args.to_console:
            proc = subprocess.Popen(
                train_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                lf.write(line)
            ret = proc.wait()
        else:
            ret = subprocess.call(train_cmd, stdout=lf, stderr=lf, cwd=os.path.dirname(os.path.abspath(__file__)))
finally:
    _release_sess_lock(lock_file)
print('train.py return code:', ret)
if ret != 0:
    raise RuntimeError('Training failed with exit code %d' % ret)

best_ckpt = os.path.join(save_dir, 'checkpoint_best.pt')
if os.path.exists(best_ckpt):
    best_copy = os.path.join(task_output_dir, '%s_best.pt' % sess)
    shutil.copyfile(best_ckpt, best_copy)
    print('Best checkpoint saved to', best_copy)
