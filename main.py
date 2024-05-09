import argparse
import math
import gc

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/home/lyh/weights/hf/vicuna_v13/7B/')
parser.add_argument('--configpath', type=str, default="config.json")
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument('--tmpdir', type=str, default='0')
parser.add_argument('--outdir', type=str, default='0')
parser.add_argument('--cpdir', type=str, default='0')
parser.add_argument('--existstate', type=str, default='')
parser.add_argument('--precision', type=str, default='fp16')
args = parser.parse_args()

train_config = {
    "lr": args.lr,
    "bs": args.bs,
    'weight_decay':0.01,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": 20,
    # Depending on your data and model size, the larger the model, the higher the sample efficiency. We recommend setting it between 20-40.
    "num_warmup_steps": 200,
    "total_steps": 800000,
    "p_w": 1.0,
    "v_w": 1.0,
    "head_w": 0.1,
    "num_workers": 4,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 2048,
    # During training, truncating the training sequences means that the larger the setting, the more training data is used, and the better the effect, but it also consumes more VRAM.
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 1.0,
    "save_freq": 1
}

import os
import torch

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs,DummyScheduler

set_seed(0)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(mixed_precision=args.precision,
                          gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
                          log_with=None,
                          dispatch_batches=True,
                          split_batches=True,
                          kwargs_handleers=[ddp_kwargs])
from model.cnets import Model
from model.configs import EConfig
from typing import Any, Dict, List

from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, AutoConfig
from model.utils import create_adamw_optimizer



baseconfig = AutoConfig.from_pretrained(args.basepath)

import random
def list_files(path,shuffle=True):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    if shuffle:
        random.shuffle(datapath)
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # try:
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data['hidden_state'][:train_config["max_len"]][None, :]
        input_ids = data['input_ids'][:train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][:train_config["max_len"]][None, :]

        # except:
        #     with open("error_path.txt", "w") as file:
        #         file.write(self.data[index])
        #     print('error path',self.data[index])

        length = hidden_state.shape[1]
        # length_q = data['query_ids'].shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target
        # sample = torch.cat((data['xs'],data['xb']))
        # sample=torch.cat((self.data[index]['x'],self.data[index]['logits']))
        # label = data['y']

        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


@torch.no_grad()
def getkacc(model, data, head, max_length=5):
    hidden_states = data["hidden_states"]
    input_ids = data["input_ids"]
    # attention_mask=data["attention_mask"]
    loss_mask = data["loss_mask"]
    # sample_mask=data["sample_mask"]
    target = data["target"]
    total = [0 for _ in range(max_length)]
    correct = [0 for _ in range(max_length)]
    bs, sl = hidden_states.shape[0], hidden_states.shape[1]
    target_headout = head(target)
    hidden_states_headout = head(hidden_states)

    for i in range(bs):
        for j in range(sl):

            single_hidden_states = hidden_states[i, :j]
            single_input_ids = input_ids[i, :j]

            single_hidden_states = single_hidden_states[None, :, :]
            single_input_ids = single_input_ids[None, :]
            for k in range(max_length):
                if loss_mask[i, single_hidden_states.shape[1] - 1] == 0:
                    break
                tmp_in_target_headout = hidden_states_headout[i, single_hidden_states.shape[1] - 1]
                tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1] - 1]
                target_in_token = torch.argmax(tmp_in_target_headout)
                target_out_token = torch.argmax(tmp_out_target_headout)
                tmp_token = input_ids[i, single_hidden_states.shape[1] - 1]
                # tmp_sample_mask=sample_mask[i,single_hidden_states.shape[1]-1]
                if not (target_in_token == tmp_token):
                    break
                out_hidden = model(single_hidden_states, input_ids=single_input_ids)
                last_hidden = out_hidden[:, -1]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout)
                total[k] += 1
                if token == target_out_token:
                    correct[k] += 1
                else:
                    for kk in range(k + 1, max_length):
                        total[kk] += 1
                    break

                single_hidden_states = torch.cat((single_hidden_states, out_hidden[:, -1:]), dim=1)
                single_input_ids = torch.cat((single_input_ids, torch.tensor([[token]]).to(single_input_ids.device)),
                                             dim=1)

    acc = [correct[i] / (total[i]+1e-8) for i in range(len(correct))]
    return acc


if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    aug = None

with accelerator.main_process_first():
    datapath = list_files(train_config["datapath"])

    traindatapath = datapath[:int(len(datapath) * 0.95)]
    testdatapath = datapath[int(len(datapath) * 0.95):]
    # print('td',train_config["datapath"])
    # print(datapath)
    # exit()
    traindataset = CustomDataset(traindatapath, transform=aug)
    testdataset = CustomDataset(testdatapath)
    train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,drop_last=True,
                              collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                              pin_memory=True)
    test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                             collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=False)



if accelerator.is_main_process:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

config = EConfig.from_pretrained(train_config["config_path"])

num_epochs = train_config["num_epochs"]
total_steps = len(train_loader)*num_epochs
is_warmup = train_config["is_warmup"]
if train_config["num_warmup_steps"] < 1:
    num_warmup_steps = train_config["num_warmup_steps"]*total_steps
else:
    num_warmup_steps = train_config["num_warmup_steps"]

model = Model(config, load_emb=True, path=args.basepath,num_res_layer=1)

criterion = nn.SmoothL1Loss(reduction="none")
# Optimizer adn Scheduler
if (
        accelerator.state.deepspeed_plugin is None
            or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        optimizer = create_adamw_optimizer(model, lr=train_config['lr'],
                                           weight_decay=train_config['weight_decay'],betas=(train_config["b1"],train_config["b2"]))
else:
    optimizer = create_adamw_optimizer(model, lr=train_config['lr'], weight_decay=train_config['weight_decay'],dummy=True)



if is_warmup and accelerator.state.deepspeed_plugin is None or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config:
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)
elif is_warmup:
    scheduler = DummyScheduler(optimizer, warmup_num_steps=int(num_warmup_steps),total_num_steps=total_steps)

if is_warmup:
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )
else:
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )
if args.existstate:
    accelerator.load_state(args.existstate)
    if accelerator.is_main_process:
        unwrap_model = accelerator.unwrap_model(model)
        torch.save({"model_state":unwrap_model.state_dict()}, f"{args.existstate}/ckpt.pt")

for epoch in range(num_epochs + 1):
    torch.cuda.empty_cache()
    gc.collect()
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    for batch_idx, data in enumerate(tqdm(train_loader)):

        with accelerator.accumulate(model):
            optimizer.zero_grad(set_to_none=True)
            if accelerator.mixed_precision=='fp16':
                inputs = data["hidden_states"].to(torch.float16)
                targets = data["target"].to(torch.float16)
            else:
                inputs = data["hidden_states"]
                targets = data["target"]
            predict = model(inputs, input_ids=data["input_ids"], attention_mask=data["attention_mask"])
            with torch.no_grad():
                target_head = model.head(targets)
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
            out_head = model.head(predict)
            out_logp = nn.LogSoftmax(dim=2)(out_head)
            loss_mask = data["loss_mask"][:, :, None]
            plogp = target_p * out_logp
            ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / loss_mask.sum()
            vloss = criterion(predict, data["target"])
            vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / loss_mask.sum()
            loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            if is_warmup and not accelerator.optimizer_step_was_skipped:
                scheduler.step()

        

        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            total += ct
            correct += cc

        del ploss, vloss
        if not math.isinf(float(loss)):
            epoch_loss += float(loss)
            num_batches += 1

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    accelerator.print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
    accelerator.print('Train Accuracy: {:.2f}%'.format(100 * correct / total))

    if (epoch + 1) % train_config["save_freq"]==0:
        torch.cuda.empty_cache()
        gc.collect()
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.eval()

        k_acc = [[] for i in range(5)]
        for batch_idx, data in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                if accelerator.mixed_precision == 'fp16':
                    inputs = data["hidden_states"].to(torch.float16)
                    targets = data["target"].to(torch.float16)
                else:
                    inputs = data["hidden_states"]
                    targets = data["target"]
                predict = model(inputs, input_ids=data["input_ids"],
                                attention_mask=data["attention_mask"])
                target_head = model.head(targets)
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
                out_head = model.head(predict)
                out_logp = nn.LogSoftmax(dim=2)(out_head)
                loss_mask = data["loss_mask"][:, :, None]
                plogp = target_p * out_logp
                ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / loss_mask.sum()
                vloss = criterion(predict, data["target"])
                vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / loss_mask.sum()
                loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
                _, predicted = torch.max(out_head, 2)
                _, target = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
                total += ct
                correct += cc
            del ploss, vloss
            if not math.isinf(float(loss)):
                epoch_loss += loss.item()
                num_batches += 1

        correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        correct, total = accelerator.gather_for_metrics((correct, total))
        correct, total = correct.sum().item(), total.sum().item()

        epoch_loss /= num_batches

        accelerator.print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        accelerator.print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
        accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}")
