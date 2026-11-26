"""Score generated IDs and answers from all 42 directions, paired by question."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze_codec import score

ARMS=("int8","int4","shared2")


def interval(values, groups):
    grouped=defaultdict(list)
    for value,group in zip(values,groups):
        grouped[group].append(value)
    counts=[(sum(v),len(v)) for v in grouped.values()]
    rng=random.Random(63411)
    draws=[]
    for _ in range(10000):
        sample=rng.choices(counts,k=len(counts))
        draws.append(sum(n for n,_ in sample)/sum(d for _,d in sample))
    draws.sort()
    return [round(100*draws[i],3) for i in (249,9749)]


def analyze(root):
    keys=list(json.loads(Path(__file__).with_name("models.json").read_text()))
    expected=set((s,t) for s in keys for t in keys if s!=t)
    pairs={}
    observed_ids=None
    aggregate={a:defaultdict(list) for a in ARMS}
    examples={a:[] for a in ARMS}
    changes={arm:dict(better=0,worse=0,equal_score_different_ids=0) for arm in ARMS}
    gold_changes={arm:dict(lost_exact_gold=0,gained_exact_gold=0) for arm in ARMS}
    sources={}
    natives={}
    for src in keys:
        inventory=json.loads((root/"sources"/(src+".json")).read_text())
        ids=set(inventory["files"])
        assert len(ids)==64 and (observed_ids is None or observed_ids==ids)
        observed_ids=ids
        sources[src]={arm:sum(row["bytes"][arm] for row in inventory["files"].values())
                      +(inventory["reference_bytes"] if arm=="shared2" else 0) for arm in ("bf16",*ARMS)}
    for src,tgt in sorted(expected):
        path=root/"pairs"/(src+"__"+tgt)
        entries=[json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]
        assert len(entries)==64 and {r["id"] for r in entries}==observed_ids,(src,tgt,len(entries))
        repeats=[r["repeat_bf16_equal"] for r in entries if "repeat_bf16_equal" in r]
        assert len(repeats)==4
        arm_data=defaultdict(list)
        for r in entries:
            assert (r["source"],r["target"])==(src,tgt)
            assert set(r["arms"])=={"bf16",*ARMS}
            gold=r["golds"]
            ref=r["arms"]["bf16"]
            base_f1,base_em=score(ref["text"],gold)
            native=json.loads((root/"native"/tgt/(r["id"]+".json")).read_text())
            natives.setdefault(tgt,{})[r["id"]]=score(native["text"],gold)[0]
            for arm,v in r["arms"].items():
                assert 0<len(v["ids"])<=24
                f1,em=score(v["text"],gold)
                same=v["ids"]==ref["ids"]
                arm_data[arm].append((f1,em,same,v["finish"]=="length"))
                if arm in ARMS:
                    aggregate[arm][(r["title"],r["id"])].append((f1-base_f1,same))
                    if not same:
                        outcome = "better" if f1>base_f1 else "worse" if f1<base_f1 else "equal_score_different_ids"
                        changes[arm][outcome]+=1
                        if base_em != em:
                            gold_changes[arm]["lost_exact_gold" if base_em else "gained_exact_gold"]+=1
                    if not same and abs(f1-base_f1)>1e-8:
                        examples[arm].append(dict(source=src,target=tgt,id=r["id"],question=r["question"],
                            golds=gold,bf16=ref["text"],packed=v["text"],delta=f1-base_f1))
        record={}
        for arm,vals in arm_data.items():
            f1,em,same,cap=zip(*vals)
            record[arm]=dict(f1=round(100*mean(f1),3),exact=sum(em),same=sum(same),capped=sum(cap))
        record["repeat_bf16"]={"same":sum(repeats),"n":len(repeats)}
        pairs[src+"__"+tgt]=record
    overall={}
    for arm,by_id in aggregate.items():
        assert len(by_id)==64 and all(len(v)==42 for v in by_id.values())
        delta=[mean(v[0] for v in vals) for vals in by_id.values()]
        sameness=[mean(v[1] for v in vals) for vals in by_id.values()]
        groups=[title for title,_ in by_id]
        overall[arm]=dict(same=sum(r[arm]["same"] for r in pairs.values()),n=42*64,
            mean_f1=round(mean(r[arm]["f1"] for r in pairs.values()),3),
            delta_f1=round(100*mean(delta),3),delta_cluster_ci=interval(delta,groups),
            agreement_cluster_ci=interval(sameness,groups),
            minimum_pair_same=min(r[arm]["same"] for r in pairs.values()),
            maximum_pair_same=max(r[arm]["same"] for r in pairs.values()),
            perfect_pairs=sum(r[arm]["same"]==64 for r in pairs.values()),
            changed_answers=changes[arm],gold_exact_changes=gold_changes[arm])
    native_repeat={}
    for directory in sorted((root/"repeated-native").glob("*")):
        repeat_files=sorted(directory.glob("*.json"))
        assert {p.stem for p in repeat_files}==observed_ids
        repeated=[json.loads(p.read_text()) for p in repeat_files]
        original=[json.loads((root/"native"/directory.name/p.name).read_text()) for p in repeat_files]
        assert all(a["prefix_sha256"]==b["prefix_sha256"] for a,b in zip(original,repeated))
        native_repeat[directory.name]=dict(n=len(original),same=sum(a==b for a,b in zip(original,repeated)))
    return dict(n_questions=64,n_articles=len(set(groups)),n_pairs=len(pairs),native_repeat=native_repeat,
        bf16_mean_f1=round(mean(r["bf16"]["f1"] for r in pairs.values()),3),
        native_f1={k:round(100*mean(v.values()),3) for k,v in natives.items()},
        overall=overall,pairs=pairs,source_bytes=sources,
        examples={arm:sorted(entries,key=lambda x:x["delta"])[:3] for arm,entries in examples.items()})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("results",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    data=analyze(a.results)
    a.output.write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(data["overall"],indent=2))
