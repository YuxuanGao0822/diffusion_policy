#!/usr/bin/env python3
"""One-command fail-fast Priming control research pipeline.

Phase 0: independently re-evaluate frozen Lowdim E350.
Phase 1: frozen teacher + bounded residual twin-Q RL.
Phase 2: end-to-end one-NFE Q guidance + teacher action anchor.
Phase 3: learned-transition-aware OT + anchored Q guidance.
Phase 4: independent PushT Image Stage-1 pretrain then phases 0-3.

Use --smoke to verify wiring before hours-long GPU work. Every stage
writes its own logs/checkpoints; --resume skips completed stages.
"""
import argparse
import csv
import json
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT=pathlib.Path(__file__).resolve().parent.parent
WORKER=ROOT/"scripts"/"train_priming_control.py"


def invoke(cmd,env,log_path):
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open("a",buffering=1) as log:
        log.write("\n$ "+" ".join(map(str,cmd))+"\n")
        print("\n$ "+" ".join(map(str,cmd)),flush=True)
        with subprocess.Popen(cmd,cwd=ROOT,env=env,
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              bufsize=1,text=True) as proc:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            rc=proc.wait()
        if rc:
            raise RuntimeError("Command failed with code "+str(rc)+
                               "; see "+str(log_path))


def best_image_ckpt(directory):
    candidates=list((directory/"checkpoints").glob("epoch=*-test_mean_score=*.ckpt"))
    if not candidates:
        raise FileNotFoundError("No scored image checkpoint: "+str(directory))
    def score(path):
        found=re.search(r"test_mean_score=([0-9.]+)\.ckpt$",path.name)
        if not found: return float("-inf")
        return float(found.group(1))
    return max(candidates,key=lambda f:(score(f),f.name))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1-checkpoint",required=True)
    p.add_argument("--image-checkpoint",default=None,
                   help="optional independent, pretrained Priming Image EMA checkpoint")
    p.add_argument("--out",default="data/outputs/priming_pusht/control_research")
    p.add_argument("--gpu",default="7",
                   help="physical GPU index; in process always cuda:0")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--online-steps",type=int,default=10000)
    p.add_argument("--eval-every",type=int,default=500)
    p.add_argument("--eval-episodes",type=int,default=30)
    p.add_argument("--heldout-episodes",type=int,default=50)
    p.add_argument("--image-pretrain-epochs",type=int,default=400)
    p.add_argument("--batch",type=int,default=128)
    p.add_argument("--warmup",type=int,default=256)
    p.add_argument("--actor-lr",type=float,default=1e-5)
    p.add_argument("--residual-lr",type=float,default=1e-4)
    p.add_argument("--critic-lr",type=float,default=3e-4)
    p.add_argument("--exploration",type=float,default=0.05)
    p.add_argument("--anchor",type=float,default=5.)
    p.add_argument("--ot-weight",type=float,default=0.1)
    p.add_argument("--effect-scale",type=float,default=1.0)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--resume",action="store_true",
                   help="skip completed stages, resume incomplete online stages")
    p.add_argument("--skip-image",action="store_true",
                   help="only run Lowdim phases 0-3; use to debug before full pipeline")
    args=p.parse_args()
    ckpt=pathlib.Path(args.stage1_checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        p.error("missing trusted Lowdim E350 EMA checkpoint: "+str(ckpt))
    if not (ROOT/"data/pusht/pusht_cchi_v7_replay.zarr").exists():
        p.error("missing PushT Zarr dataset under data/pusht/")
    out=pathlib.Path(args.out)
    if not out.is_absolute(): out=ROOT/out
    out=out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES":str(args.gpu),
                "CUDA_DEVICE_ORDER":"PCI_BUS_ID",
                "SDL_VIDEODRIVER":"dummy","SDL_AUDIODRIVER":"dummy",
                "WANDB_MODE":"disabled","HYDRA_FULL_ERROR":"1",
                "OMP_NUM_THREADS":"8"})
    steps=80 if args.smoke else args.online_steps
    every=40 if args.smoke else args.eval_every
    eval_eps=2 if args.smoke else args.eval_episodes
    heldout=2 if args.smoke else args.heldout_episodes
    batch=8 if args.smoke else args.batch
    warmup=8 if args.smoke else args.warmup
    manifest=out/"pipeline_manifest.json"
    def record(name,info):
        content=json.loads(manifest.read_text()) if manifest.exists() else {}
        content[name]=info
        manifest.write_text(json.dumps(content,indent=2,ensure_ascii=False))

    def run_modality(modality,teacher):
        prefix=out/modality
        stages=["baseline","residual","q","task_ot"]
        previous=None
        for idx,phase in enumerate(stages):
            work=prefix/("0_baseline" if phase=="baseline" else
                         {"residual":"1_residual","q":"2_anchored_q",
                          "task_ot":"3_task_aware_ot"}[phase])
            work.mkdir(parents=True,exist_ok=True)
            required=work/("heldout.json" if phase=="baseline" else "success.json")
            if args.resume and required.is_file():
                print("Skip finished",modality,phase,flush=True)
            else:
                cmd=[sys.executable,str(WORKER),
                     "--phase",phase,"--modality",modality,
                     "--teacher",str(teacher),"--out",str(work),
                     "--device","cuda:0","--seed",str(args.seed),
                     "--steps",str(steps),"--eval-every",str(every),
                     "--eval-episodes",str(eval_eps),
                     "--report-episodes",str(heldout),
                     "--batch",str(batch),"--warmup",str(warmup),
                     "--actor-lr",str(args.actor_lr),
                     "--residual-lr",str(args.residual_lr),
                     "--critic-lr",str(args.critic_lr),
                     "--exploration",str(args.exploration),
                     "--anchor",str(args.anchor),
                     "--ot-weight",str(args.ot_weight),
                     "--effect-scale",str(args.effect_scale)]
                if previous is not None:
                    cmd.extend(["--previous",str(previous)])
                if args.resume:
                    cmd.append("--resume")
                invoke(cmd,env,work/"terminal.log")
                if not required.is_file():
                    raise RuntimeError(phase+" did not write completion marker")
            if phase!="baseline":
                previous=work/"best.pt"
                if not previous.is_file():
                    raise FileNotFoundError("missing best actor: "+str(previous))
            info=json.loads(required.read_text())
            record(modality+"/"+phase,info)
        print(modality,"four stages complete",flush=True)

    run_modality("lowdim",ckpt)
    if args.skip_image:
        print("Lowdim pipeline complete. Image validation explicitly skipped.")
        return

    image=pathlib.Path(args.image_checkpoint).expanduser().resolve() if args.image_checkpoint else None
    if image is not None and not image.is_file():
        raise FileNotFoundError(image)
    if image is None:
        image_pretrain=out/"image_stage1_pretrain"
        complete=image_pretrain/"pretrain_complete.json"
        if args.resume and complete.is_file():
            previous=json.loads(complete.read_text())
            image=pathlib.Path(previous["checkpoint"]).resolve()
            if not image.is_file():
                raise FileNotFoundError("completed Image teacher missing: "+str(image))
        elif args.resume and image_pretrain.exists() and any(image_pretrain.iterdir()):
            raise RuntimeError(
                "Image pretraining is incomplete: "+str(image_pretrain)+
                ". Refusing to treat a partial Stage-1 run as a completed "
                "teacher. Finish/resume it manually or start a NEW --out dir.")
        if image is None:
            cmd=[sys.executable,"train.py",
                 "--config-name=train_priming_image_workspace",
                 "training.device=cuda:0",
                 "training.seed="+str(args.seed),
                 "training.resume=false",
                 "training.num_epochs="+str(2 if args.smoke else args.image_pretrain_epochs),
                 "training.debug="+("true" if args.smoke else "false"),
                 "training.rollout_every="+str(1 if args.smoke else 25),
                 "training.checkpoint_every="+str(1 if args.smoke else 25),
                 "training.val_every="+str(1 if args.smoke else 10),
                 "task.env_runner.n_train_vis=0",
                 "task.env_runner.n_test_vis=0",
                 "task.env_runner.n_test="+str(2 if args.smoke else 50),
                 "logging.mode=disabled",
                 "hydra.run.dir="+str(image_pretrain)]
            invoke(cmd,env,image_pretrain/"terminal.log")
            image=best_image_ckpt(image_pretrain)
            complete.write_text(json.dumps({
                "checkpoint":str(image),"epochs":2 if args.smoke else args.image_pretrain_epochs,
                "finished_at":time.time()},indent=2))
        record("image/stage1_teacher",{"path":str(image)})
    run_modality("image",image)
    report=json.loads(manifest.read_text())
    summary=[]
    for modality in ("lowdim","image"):
        base=report[modality+"/baseline"]["mean_score"]
        for phase in ("baseline","residual","q","task_ot"):
            result=report[modality+"/"+phase]
            mean=(result["mean_score"] if phase=="baseline"
                  else result["heldout_mean"])
            summary.append({"modality":modality,"phase":phase,
                "heldout_mean_score":mean,
                "delta_from_frozen_teacher":mean-base,
                "selection_mean":(result["mean_score"] if phase=="baseline"
                     else result["best_selection_mean"])})
    with (out/"summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    markdown=[
        "# Priming Control: Held-out Results",
        "",
        "These are empirical scores, not statistical significance tests.",
        "Selection seeds 220000+, held-out seeds 320000+.",
        "",
        "| Modality | Stage | Held-out score | Delta vs frozen |",
        "|---|---|---:|---:|"]
    for row in summary:
        markdown.append(
            "| "+row["modality"]+" | "+row["phase"]+" | "+
            format(row["heldout_mean_score"],".5f")+" | "+
            format(row["delta_from_frozen_teacher"],"+.5f")+" |")
    (out/"summary.md").write_text("\n".join(markdown)+"\n")
    print("ALL PHASES COMPLETE:",manifest,flush=True)
    print("Research summary:",out/"summary.md",flush=True)


if __name__=="__main__":
    main()
