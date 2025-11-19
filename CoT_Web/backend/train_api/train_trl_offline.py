#!/usr/bin/env python3
import os, json, argparse, math, random
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

# ---------------------------
# GPU 메모리 자동 배분 유틸
# ---------------------------
def _visible_gpu_indices() -> List[int]:
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))

def _max_memory_from_free(gpu_idxs: List[int], safety_gb: float = 1.0) -> Dict[int, str]:
    max_mem = {}
    for i in gpu_idxs:
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(i)
            free_gb = free_bytes / (1024 ** 3)
            alloc_gb = max(1.0, free_gb - safety_gb)  # 최소 1GiB은 남기고 사용
            max_mem[i] = f"{int(alloc_gb)}GiB"
        except Exception:
            # mem_get_info가 실패해도 안전하게 기본값 제공
            max_mem[i] = "10GiB"
    if not gpu_idxs:
        # CPU용 기본값
        max_mem["cpu"] = "64GiB"
    return max_mem

# ---------------------------
# DPO 데이터셋
# ---------------------------
class DpoPairs(Dataset):
    def __init__(self, path: str, processor, max_length: int):
        self.rows=[]
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                obj=json.loads(line)
                if all(k in obj for k in ["prompt","chosen","rejected"]):
                    self.rows.append({"prompt":obj["prompt"],"chosen":obj["chosen"],"rejected":obj["rejected"]})
        if not self.rows:
            raise RuntimeError(f"No samples in {path}")
        self.processor=processor
        self.tok=processor.tokenizer
        self.max_length=max_length

    def _chat(self, prompt: str, answer: str) -> str:
        msgs=[{"role":"user","content":[{"type":"text","text":prompt}]},{"role":"assistant","content":[{"type":"text","text":answer}]}]
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    def _prompt_only(self, prompt: str) -> str:
        msgs=[{"role":"user","content":[{"type":"text","text":prompt}]}]
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def __getitem__(self, idx):
        r=self.rows[idx]
        pc=self._prompt_only(r["prompt"])
        fc=self._chat(r["prompt"], r["chosen"])
        fr=self._chat(r["prompt"], r["rejected"])

        enc_p=self.tok(pc, truncation=True, max_length=self.max_length, add_special_tokens=False)
        enc_c=self.tok(fc, truncation=True, max_length=self.max_length, add_special_tokens=False)
        enc_r=self.tok(fr, truncation=True, max_length=self.max_length, add_special_tokens=False)

        pl=len(enc_p["input_ids"])

        ids_c=torch.tensor(enc_c["input_ids"], dtype=torch.long)
        msk_c=torch.tensor(enc_c["attention_mask"], dtype=torch.long)
        ansmask_c=torch.zeros_like(msk_c); ansmask_c[pl:]=1

        ids_r=torch.tensor(enc_r["input_ids"], dtype=torch.long)
        msk_r=torch.tensor(enc_r["attention_mask"], dtype=torch.long)
        ansmask_r=torch.zeros_like(msk_r); ansmask_r[pl:]=1

        return {
            "input_ids_chosen":ids_c, "attention_mask_chosen":msk_c, "ansmask_chosen":ansmask_c,
            "input_ids_rejected":ids_r, "attention_mask_rejected":msk_r, "ansmask_rejected":ansmask_r
        }

    def __len__(self):
        return len(self.rows)

def collate_fn(batch: List[Dict[str,Any]], pad_id: int):
    max_len_c=max(x["input_ids_chosen"].size(0) for x in batch)
    max_len_r=max(x["input_ids_rejected"].size(0) for x in batch)

    def pad_stack(key, max_len):
        out=[]
        for x in batch:
            t=x[key]
            if t.size(0)<max_len:
                pad=torch.full((max_len-t.size(0),), pad_id if "input_ids" in key else 0, dtype=t.dtype)
                t=torch.cat([t,pad],dim=0)
            out.append(t.unsqueeze(0))
        return torch.cat(out,dim=0)

    ids_c=pad_stack("input_ids_chosen",max_len_c)
    msk_c=pad_stack("attention_mask_chosen",max_len_c)
    am_c =pad_stack("ansmask_chosen",max_len_c)

    ids_r=pad_stack("input_ids_rejected",max_len_r)
    msk_r=pad_stack("attention_mask_rejected",max_len_r)
    am_r =pad_stack("ansmask_rejected",max_len_r)

    return {
        "input_ids_chosen":ids_c, "attention_mask_chosen":msk_c, "ansmask_chosen":am_c,
        "input_ids_rejected":ids_r, "attention_mask_rejected":msk_r, "ansmask_rejected":am_r
    }

def compute_logprob_per_token(model, input_ids, attention_mask):
    outputs=model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits=outputs.logits[:,:-1,:]
    labels=input_ids[:,1:]
    am=attention_mask[:,1:]
    logp=torch.log_softmax(logits,dim=-1)
    gather=logp.gather(-1,labels.unsqueeze(-1)).squeeze(-1)
    return gather*am

def dpo_loss(beta, lp_chosen, lp_rejected, ansmask_chosen, ansmask_rejected):
    tl_c=(ansmask_chosen[:,1:]).sum(dim=1).clamp(min=1)
    tl_r=(ansmask_rejected[:,1:]).sum(dim=1).clamp(min=1)
    lp_c=(lp_chosen*(ansmask_chosen[:,1:])).sum(dim=1)/tl_c
    lp_r=(lp_rejected*(ansmask_rejected[:,1:])).sum(dim=1)/tl_r
    diff=lp_c-lp_r
    return -torch.nn.functional.logsigmoid(beta*diff).mean()

def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base_model",type=str,required=True)
    ap.add_argument("--dpo_jsonl",type=str,required=True)
    ap.add_argument("--out_dir",type=str,required=True)
    ap.add_argument("--epochs",type=int,default=100)
    ap.add_argument("--bsz",type=int,default=1)
    ap.add_argument("--grad_accum",type=int,default=1)
    ap.add_argument("--lr",type=float,default=2e-5)
    ap.add_argument("--beta",type=float,default=0.1)
    ap.add_argument("--max_length",type=int,default=2048)
    ap.add_argument("--bf16",action="store_true")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--target_modules",type=str,default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    # 선택: GPU 안전 여유 메모리(GB) 잔여
    ap.add_argument("--safety_gb",type=float,default=1.0)
    return ap.parse_args()

def set_seed(seed:int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    args=parse_args()
    os.makedirs(args.out_dir,exist_ok=True)
    set_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    gpu_idxs = _visible_gpu_indices()
    max_memory = _max_memory_from_free(gpu_idxs, safety_gb=args.safety_gb)

    # 속도/안정성 세팅
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # dtype 결정
    if use_cuda:
        dtype = torch.bfloat16 if args.bf16 else torch.float16
    else:
        dtype = torch.float32

    # 프로세서
    processor = AutoProcessor.from_pretrained(args.base_model, use_fast=True)

    # ---------------------------
    # 모델 로딩: device_map="auto" + max_memory (여유 메모리 기반)
    # ---------------------------
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=("auto" if use_cuda else None),
        max_memory=max_memory if use_cuda else None
    )

    # LoRA 적용
    lconf=LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        task_type="CAUSAL_LM"
    )
    model=get_peft_model(model,lconf)
    model.train()

    # 데이터/로더
    ds=DpoPairs(args.dpo_jsonl,processor,args.max_length)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    dl=DataLoader(ds,batch_size=args.bsz,shuffle=True,
                  collate_fn=lambda b: collate_fn(b, pad_id))

    opt=AdamW(model.parameters(),lr=args.lr)
    scaler=torch.cuda.amp.GradScaler(enabled=(dtype in (torch.float16, torch.bfloat16) and use_cuda))

    steps_per_epoch=math.ceil(len(ds)/max(1,args.bsz))
    total_steps=args.epochs*steps_per_epoch
    global_step=0

    for _ in range(args.epochs):
        for batch in dl:
            # device_map='auto' 이므로 텐서는 모델이 있는 디바이스에 자동 전송되지 않음
            # → 입력 텐서는 모델 첫 번째 디바이스로 보냄 (Qwen2.5는 내부에서 dispatch)
            first_device = next(model.parameters()).device
            for k in batch: batch[k]=batch[k].to(first_device, non_blocking=True)

            # autocast
            if use_cuda and dtype in (torch.float16, torch.bfloat16):
                ctx = torch.autocast(device_type="cuda", dtype=dtype)
            else:
                # CPU 학습
                class _dummy: 
                    def __enter__(self): return None
                    def __exit__(self,*a): return False
                ctx = _dummy()

            with ctx:
                lp_c=compute_logprob_per_token(model,batch["input_ids_chosen"],batch["attention_mask_chosen"])
                lp_r=compute_logprob_per_token(model,batch["input_ids_rejected"],batch["attention_mask_rejected"])
                loss=dpo_loss(args.beta,lp_c,lp_r,batch["ansmask_chosen"],batch["ansmask_rejected"])
                loss=loss/max(1,args.grad_accum)

            if use_cuda and dtype==torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step+1)%max(1,args.grad_accum)==0:
                if use_cuda and dtype==torch.float16:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
            global_step+=1

    # 저장
    model.save_pretrained(args.out_dir)
    try: processor.save_pretrained(args.out_dir)
    except: pass
    print(f"[OK] DPO LoRA saved to: {args.out_dir}")

if __name__=="__main__":
    main()
