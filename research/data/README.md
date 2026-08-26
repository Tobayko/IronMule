# Evaluation data

The quality experiment (E13) used **SQuAD v1.1 dev**, which is licensed CC BY-SA 4.0
and is therefore not redistributed here. Fetch it if you want to reproduce E13:

```bash
curl -sSL -o research/data/squad-dev-v1.1.json \
  https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
shasum -a 256 research/data/squad-dev-v1.1.json
# expected: 95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9
```

Rajpurkar et al., *SQuAD: 100,000+ Questions for Machine Comprehension of Text*, 2016.

Experiments E14 to E16 also read this file for realistic prompt material.
