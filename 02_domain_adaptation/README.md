# Domain adaptation: CORAL + PCA/LDA

This section aligns the 256-dimensional bottleneck embeddings from Hospital A
and Hospital B and visualizes the learned domain gap.

The training integration lives in `../01_model_federated/src/domain_adaptation.py`
because that is where the model and federated loop are executed.  CORAL runs
after each FedAvg aggregation when `use_domain_adaptation=True`.  The physical
volume batch remains 1; a detached rolling queue provides prior embeddings for
the covariance estimate without retaining multiple 3D computation graphs.

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

