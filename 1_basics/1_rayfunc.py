import ray
ray.init()

@ray.remote
def zero():
    return 0

ids = []
for _ in range(4):
    out_id = zero.remote()
    ids.append(out_id)

[print(ray.get(id)) for id in ids]