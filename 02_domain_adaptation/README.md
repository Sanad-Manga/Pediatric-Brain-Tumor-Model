# Domain adaptation: CORAL + PCA/LDA

This section aligns the 256-dimensional bottleneck embeddings from Hospital A
and Hospital B and visualizes the learned domain gap.

The training integration lives in `../01_model_federated/src/domain_adaptation.py`
because that is where the model and federated loop are executed.  CORAL runs
after each FedAvg aggregation when `use_domain_adaptation=True`.  The physical
volume batch remains 1; a detached rolling queue provides prior embeddings for
the covariance estimate without retaining multiple 3D computation graphs.

## Domain-only tests

```bash
cd ../01_model_federated
python -m pytest tests/test_domain_adaptation.py -q

cd ../02_domain_adaptation
python -m pytest tests -q
```

## Visualize before vs. after

```bash
python visualize_features.py \
  --cache-path /path/to/cache_96cube \
  --hospital-a ../00_shared/manifests/hospitalA.json \
  --hospital-b ../00_shared/manifests/hospitalB.json \
  --heldout ../00_shared/manifests/heldout.json \
  --before-checkpoint /path/to/baseline.pt \
  --after-checkpoint /path/to/coral.pt \
  --output feature_domains.png
```

The output contains PCA and LDA projections for both checkpoints. LDA uses the
institution label only for visualization; it is not used by model training.

Passing tests and a finite positive CORAL smoke loss establish technical
correctness only. They do not show that tumor segmentation improved. Any
performance claim requires a controlled baseline-versus-CORAL comparison using
held-out ET, NC, WT, and mean Dice scores.

## Real-cache smoke validation

The branch was exercised on a Colab GPU with two real Hospital A subjects and
two real Hospital B subjects from the 96³ cache:

- all 227 cached subjects were discovered;
- one federated round completed with physical batch size 1;
- segmentation losses were finite;
- the checkpoint was saved successfully; and
- checkpoint CORAL loss was finite and positive (`5.006667969099e-09`).

The smoke run validates execution and checkpoint wiring. It is not an
effectiveness result; queue sizes of two are intentionally minimal and full
baseline-versus-CORAL training remains required.
