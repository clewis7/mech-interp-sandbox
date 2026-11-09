import json
from pathlib import Path

import nltk
import torch
from nltk.corpus import wordnet as wn

# ---- paths ----
BUNDLE_PATH = Path("replay_bundle.pt")
OUT_PATH = Path("class_groups_dogsplit.json")

# ---- groups: (name, WordNet synsets), ordered SPECIFIC -> GENERAL ----
# a class goes to the FIRST group whose synset is among its ancestors, so order matters
GROUPS = [
    ("terrier", ["terrier.n.01"]),
    ("hound", ["hound.n.01"]),
    ("sporting dog", ["sporting_dog.n.01"]),
    ("working dog", ["working_dog.n.01"]),
    ("toy dog", ["toy_dog.n.01"]),
    ("other dog", ["dog.n.01"]),
    ("wild canine", ["canine.n.02"]),
    ("cat", ["feline.n.01"]),
    ("bear", ["bear.n.01"]),
    ("primate", ["primate.n.02"]),
    ("other mammal", ["mammal.n.01"]),
    ("bird", ["bird.n.01"]),
    ("reptile & amphibian", ["reptile.n.01", "amphibian.n.03"]),
    ("fish", ["fish.n.01"]),
    ("insect", ["insect.n.01"]),
    ("other invertebrate", ["invertebrate.n.01"]),
    ("fungus", ["fungus.n.01"]),
    ("fruit & vegetable", ["edible_fruit.n.01", "vegetable.n.01"]),
    ("food & dish", ["food.n.01", "food.n.02"]),
    ("plant & seed", ["plant.n.02", "plant_part.n.01"]),
    ("person", ["person.n.01"]),
    ("landscape", ["geological_formation.n.01"]),
    ("aircraft", ["aircraft.n.01"]),
    ("boat", ["vessel.n.02", "boat.n.01"]),
    ("land vehicle", ["self-propelled_vehicle.n.01", "wheeled_vehicle.n.01"]),
    ("other vehicle", ["vehicle.n.01", "conveyance.n.03"]),
    ("musical instrument", ["musical_instrument.n.01"]),
    ("balls & sports gear", ["ball.n.01", "game_equipment.n.01", "sports_equipment.n.01"]),
    ("clothing", ["clothing.n.01"]),
    ("furniture", ["furniture.n.01"]),
    ("electronics", ["electronic_equipment.n.01", "computer.n.01"]),
    ("weapon", ["weapon.n.01"]),
    ("home appliance", ["home_appliance.n.01"]),
    ("kitchenware", ["kitchen_utensil.n.01", "cooking_utensil.n.01"]),
    ("container", ["container.n.01"]),
    ("tool & implement", ["tool.n.01", "implement.n.01"]),
    ("structure & building", ["structure.n.01"]),
    ("machine & device", ["machine.n.01", "device.n.01"]),
    ("other object", ["artifact.n.01", "natural_object.n.01"]),
]
OTHER = "other"
MANUAL = {"rapeseed": "landscape"}  # WordNet calls rapeseed a seed; its images are yellow fields
CHECK = ["rapeseed", "valley", "tennis_ball", "Granny_Smith", "broccoli", "Norwich_terrier", "cab", "ladybug", "bee"]

nltk.download("wordnet", quiet=True)

# ---- classes from the bundle (wnids are WordNet 3.0 offsets, which nltk's wordnet uses) ----
bundle = torch.load(BUNDLE_PATH)
wnids = bundle["wnids"]
class_names = bundle["class_names"]
probe_labels = bundle["probe_labels"].long()

# ---- resolve group synsets (warn and skip any name that doesn't exist) ----
group_names = [name for name, _ in GROUPS] + [OTHER]
group_syns = []
for name, syn_names in GROUPS:
    syns = set()
    for s in syn_names:
        try:
            syns.add(wn.synset(s))
        except Exception:
            print(f"  warning: synset {s} not found (group '{name}')")
    group_syns.append(syns)

# ---- assign each class: walk all hypernym paths, take the first matching group ----
class_to_group = []
for wnid in wnids:
    syn = wn.synset_from_pos_and_offset("n", int(wnid[1:]))
    anc = {syn} | set(syn.closure(lambda s: s.hypernyms() + s.instance_hypernyms()))
    class_to_group.append(next((i for i, syns in enumerate(group_syns) if syns & anc), len(GROUPS)))
for cname, gname in MANUAL.items():
    class_to_group[class_names.index(cname)] = group_names.index(gname)

# ---- report ----
group_labels = torch.tensor(class_to_group)[probe_labels]
print(f"\n{'group':<22} {'classes':>7} {'probe imgs':>10}  examples")
for g, name in enumerate(group_names):
    members = [class_names[c] for c in range(len(wnids)) if class_to_group[c] == g]
    print(f"{name:<22} {len(members):>7} {(group_labels == g).sum().item():>10}  {', '.join(members[:5])}")

print("\narticle classes:")
for c in CHECK:
    if c in class_names:
        print(f"  {c:<18} -> {group_names[class_to_group[class_names.index(c)]]}")

others = [class_names[c] for c in range(len(wnids)) if class_to_group[c] == len(GROUPS)]
print(f"\n'{OTHER}' ({len(others)} classes): {', '.join(others)}")

# ---- save ----
OUT_PATH.write_text(json.dumps({"groups": group_names, "class_to_group": class_to_group}, indent=1))
print(f"\nsaved {OUT_PATH}")