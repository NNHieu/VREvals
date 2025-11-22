import io
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.pool import ThreadPool
from typing import Any, Callable
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio # Alias for clarity
import asyncio

def map_with_progress(
    f: Callable,
    xs: list[Any],
    num_threads: int = os.cpu_count() or 10,
    pbar: bool = True,
):
    """
    Apply f to each element of xs, using a ThreadPool, and show progress.
    """
    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x

    if os.getenv("debug"):
        return list(map(f, pbar_fn(xs, total=len(xs))))
    else:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            return list(pbar_fn(pool.imap(f, xs), total=len(xs)))

def gather_with_progress(
    f: Callable,
    xs: list[Any],
    num_threads: int = os.cpu_count() or 10,
    pbar: bool = True,
):
    """
    Asynchronously apply f to each element of xs using asyncio.gather, with a progress bar.
    """
    semaphore = asyncio.Semaphore(min(num_threads, len(xs)))

    async def wrapper():
        async def run_with_semaphore(semaphore, x):
            async with semaphore:
                return await f(x)
        # Create coroutine tasks for each item in xs
        tasks = [asyncio.create_task(run_with_semaphore(semaphore,x)) for x in xs]
        results = []
        if pbar:
            # for future in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            #     res = await future
            #     results.append(res)
            results = await tqdm_asyncio.gather(*tasks, desc="Processing")
        else:
            results = await asyncio.gather(*tasks)
        # If used with progress bar, results may be out of order, fix back to original order
        # if pbar:
        #     # There is no guarantee that order matches xs, so build from tasks
        #     task_to_idx = {task: i for i, task in enumerate(tasks)}
        #     ordered_results = [None] * len(xs)
        #     for task, res in zip(tasks, results):
        #         idx = task_to_idx[task]
        #         ordered_results[idx] = res
        #     return ordered_results
        return results

    # Gather can only be called from an async context, so check and run accordingly
    try:
        loop = asyncio.get_running_loop()
        # Already in event loop; return coroutine (let caller await)
        return wrapper()
    except RuntimeError:
        # Not in event loop; run it
        return asyncio.run(wrapper())

def model_path_to_short_name(model_path: str):
    if 'qwq' in model_path.lower():
        model_short_name = 'qwq'
    elif 'deepseek' in model_path.lower():
        if 'llama-8b' in model_path.lower():
            model_short_name = 'ds-llama-8b'
        elif 'qwen-7b' in model_path.lower():
            model_short_name = 'ds-qwen-7b'
        elif 'qwen-32b' in model_path.lower():
            model_short_name = 'ds-qwen-32b'
    elif 'sky-t1' in model_path.lower():
        model_short_name = 'sky-t1'
    elif 'qwen-math' in model_path or 'ds-qwen' in model_path:
        model_short_name = model_path
    else:
        model_short_name = model_path.split('/')[-1].lower().replace('-instruct', '')
    
    return model_short_name
