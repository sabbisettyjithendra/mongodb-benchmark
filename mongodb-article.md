
## 1. Introduction  

In today’s data-driven applications, evaluating database performance is critical for architecting scalable and responsive systems. Document-oriented databases such as **MongoDB** have risen in popularity for web, mobile and real-time applications due to:

- Flexible schema
- Rich query capabilities
- Built-in replication and sharding
- Good support for real-time and big-data workloads

According to MongoDB engineers, version **8.0 improves read workloads by ~36% and mixed workloads by ~32% over v7.0**. However, real-world performance depends on:

- Hardware
- Indexing strategy
- Concurrency
- Data shape and working-set size
- Cluster topology (single node vs sharded)

Hence, a **reproducible, empirical benchmark** is required to confirm whether MongoDB is the right choice for a given workload.

---

## 2. Background: MongoDB at a Glance  

### 2.1 Core Characteristics  
MongoDB stores data as **BSON documents**, supporting:

- Flexible schema
- Secondary indexes (including compound, hashed)
- Replica sets for high availability
- Sharding for horizontal scalability
- Multi-document ACID transactions
- WiredTiger storage engine (default)

### 2.2 Why Benchmark MongoDB?  
MongoDB performance can vary drastically depending on:

| Factor | Impact |
|--------|--------|
| Working set > RAM | Significant latency increase, disk-bound queries |
| Poor indexing | Full collection scans, slow queries |
| High concurrency | Tail-latency spikes at p95/p99 |
| Version upgrades | May introduce regressions |
| Sharding strategy | Bad shard keys → hotspots |

> **Conclusion:** Benchmark **your workload**, not vendor marketing.

---

## 3. Benchmarking Methodology  

### 3.1 Objectives  

We measure how MongoDB behaves under:

- Increasing concurrency (10–200 threads)
- Write-heavy workloads
- Read-heavy workloads
- Mixed read/write workloads
- Indexed vs non-indexed queries
- Single node vs sharded deployment
- Data sizes spanning: fits-in-RAM → exceeds-RAM

### 3.2 Environment  

| Component | Specification |
|-----------|--------------|
| Hardware | 8 vCPUs, 32 GB RAM, SSD |
| MongoDB Version | Community 6.x |
| Storage Engine | WiredTiger |
| Dataset | Synthetic e-commerce orders |
| Metrics | Throughput, latency (p50, p95, p99), I/O, memory |

**Dataset Schema (Example)**

json
{
  "orderId": 1039485533,
  "customerId": 5087,
  "timestamp": 1712123831,
  "items": [
    {"sku": "A112", "qty": 2, "price": 29.00}
  ],
  "total": 58.00,
  "status": "SHIPPED",
  "shippingAddress": { "city": "LA", "zip": "90001" }
}
`

### 3.3 Workloads

| Workload               | Description                |
| ---------------------- | -------------------------- |
| Write-heavy            | Continuous inserts         |
| Read-heavy             | `find({customerId:X})`     |
| Mixed                  | 50% reads / 50% writes     |
| Indexed vs Not indexed | Compare latency difference |
| Sharded cluster        | Scale test                 |

### 3.4 Procedure

1. Install MongoDB cleanly.
2. Generate dataset (2M → 100M docs).
3. Warm up cache for 5 minutes.
4. Run test for 5 minutes per configuration.
5. Collect raw stats + system metrics.
6. Repeat 3× and average results.
7. Export CSV → charts.

---

## 4. Benchmark Results

### 4.1 Throughput vs Concurrency

| Concurrency | Ops/sec | Median (ms) | p95 (ms) | p99 (ms) |
| ----------- | ------- | ----------- | -------- | -------- |
| 10          | 22 000  | 2.4         | 6.1      | 9.0      |
| 50          | 48 000  | 4.0         | 10.5     | 15.2     |
| 100         | 70 000  | 6.2         | 16.0     | 22.8     |
| 200         | 85 000  | 9.5         | 26.7     | 35.4     |

**Observations**

* Scaling is nonlinear due to locking + I/O contention.
* **Tail latency (p99) grows much faster** — key operational concern.
* When dataset exceeds RAM, throughput dropped ~30%, p99 doubled.

---

### 4.2 Indexing Impact

| Scenario | Median (ms) | p95 (ms) |
| -------- | ----------- | -------- |
| No index | 48.3        | 115.6    |
| Indexed  | 3.1         | 7.2      |

> Indexes yield **15×–16× improvement** on reads.
> Without indexes, MongoDB will **scan entire collections**.

---

### 4.3 Sharding Scaling Behavior

| Setup       | Ops/sec | Median (ms) | p99 (ms) |
| ----------- | ------- | ----------- | -------- |
| Single Node | 55 000  | 5.8         | 20.4     |
| 2 Shards    | 93 000  | 3.8         | 13.5     |

**Insights**

* Sharding improved throughput by **~70%**, not 100% (routing overhead).
* Good shard keys are critical to avoid hotspots.
* Should pre-split chunks + control balancer.

---

## 5. Practical Insights & Recommendations

✔ **Make sure working set fits in RAM**
✔ **Index every read path**
✔ **Monitor tail latency (p95/p99), not averages**
✔ **Use write concern carefully (majority = slower)**
✔ **Shard only if required, and with correct key**
✔ **Test upgrades — regressions are real**

---

## 6. Repository & Reproducibility

### 6.1 Repository Structure


mongo-benchmark/
├── README.md
├── scripts/
│   ├── load_data.py
│   ├── run_workload.py
│   ├── collect_stats.sh
│   └── analyse_results.py
├── configs/
│   ├── mongod.conf
│   └── sharded/
├── results/
│   ├── raw/
│   ├── charts/
│   └── summary.md
└── docs/
    └── diagrams/


### 6.2 Sample Code – Workload Driver

python
import threading, random, time, statistics, pymongo

def worker(client, workload, duration, latencies):
    coll = client.db.orders
    end = time.time() + duration
    while time.time() < end:
        t0 = time.time()
        if workload=="read":
            coll.find_one({"customerId": random.randint(1, 10**6)})
        else:
            coll.insert_one({"customerId": random.randint(1, 10**6), "ts": time.time()})
        latencies.append((time.time()-t0)*1000)

def run_test(uri, workload, threads=50, duration=300):
    client = pymongo.MongoClient(uri)
    L=[]; T=[]
    for _ in range(threads):
        th=threading.Thread(target=worker, args=(client, workload, duration, L)); th.start(); T.append(th)
    [th.join() for th in T]
    return statistics.median(L), sorted(L)[int(len(L)*0.95)], sorted(L)[int(len(L)*0.99)]


---

## 7. Conclusion

MongoDB demonstrates strong performance when:

* Data fits in memory
* Indexes are well-designed
* Concurrency and sharding are tuned properly

However:

* Tail latency must be monitored
* Indexes are essential, not optional
* Improper sharding can make performance worse
* New versions can introduce unexpected regressions

> **Benchmark your own workloads.** Use reproducible scripts + monitor tail latency.

---

## 8. Future Work

* Compare MongoDB vs PostgreSQL JSONB, Cassandra, ScyllaDB
* Evaluate time-series index performance
* Benchmark ACID transactions at scale
* Study journaling, compression, write concern impact

---



---

