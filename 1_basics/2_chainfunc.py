import ray

@ray.remote
def zero():
    return 0

@ray.remote
def inc(x):
    return x + 1

ray.init()

result_id = inc.remote(zero.remote())
print(ray.get(result_id))