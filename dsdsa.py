def main():
    check_real_gpu_count()
    recover_stale_gpus()
    while True:
        item = r.blpop("job_queue", timeout=5)
        if not item:
            continue
        _, job_id = item
        job = json.loads(r.get(f"job:{job_id}") or "{}")
        if job.get("status") == "cancelled":
            continue   # задачу скасовано до виконання
        gpu_id = None
        while gpu_id is None:
            if get_job(job_id).get("cancel_requested"):
                break
            gpu_id = get_free_gpu()
            if gpu_id is None:
                time.sleep(1)
        if gpu_id is None:
            continue
        t = threading.Thread(
            target=execute_job, args=(job_id, gpu_id), daemon=True)
        t.start()
