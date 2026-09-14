import time, torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

def train_epoch(model, loader, optimizer, loss_fn, device="cuda",
                amp=True, log_interval=50, grad_accum_steps=1):
    model.train(); scaler = GradScaler(enabled=amp)
    avg = []; t0=time.time()
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(tqdm(loader, desc="train", ncols=100)):
        hazy, clear = batch["hazy"].to(device), batch["clear"].to(device)
        with autocast(enabled=amp):
            out = model(hazy)
            loss, logs = loss_fn(hazy, out["clean"], clear)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % grad_accum_steps == 0:
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg.append(loss.item() * grad_accum_steps)  # 记录还原到未除前
        if (i+1) % log_interval == 0:
            print(f"[{i+1}/{len(loader)}] loss={sum(avg)/len(avg):.4f} | {logs}")
    return sum(avg)/len(avg), time.time()-t0

@torch.no_grad()
def validate(model, loader, loss_fn, device="cuda"):
    model.eval(); avg=[]
    for batch in tqdm(loader, desc="val", ncols=100):
        hazy, clear = batch["hazy"].to(device), batch["clear"].to(device)
        out = model(hazy)
        loss, _ = loss_fn(hazy, out["clean"], clear)
        avg.append(loss.item())
    return sum(avg)/len(avg)
