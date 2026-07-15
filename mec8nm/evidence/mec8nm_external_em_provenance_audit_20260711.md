# External EM Asset Provenance Audit

Updated: 2026-07-11T14:10Z

## Scope

This is a provenance and suitability audit for the inherited external EM assets
under `/volume/med-train/users/dchen02/code/dbMiM/data/EM_pretrain_data/` and
the small public raw-data cache. It does not change the supervised MEC split:
M01--07/M09 remain train, M08/M10 remain validation, and M11/M12 remain frozen.

## Local Evidence

- `public_em_manifest.json` records ISBI 2012 and SNEMI3D raw inputs and the
  existing TOS cache prefix, but it records only short source names rather than
  upstream release URLs, version identifiers, checksums, or licenses.
- `fafb_manifest.json`, `fib25_manifest.json`, `kasthuri_manifest.json`,
  `mitoem_manifest.json`, and `mb_moc_manifest.json` identify
  `cyd0806/EM_pretrain_data` as a distribution mirror. The manifests preserve
  mirror object identities and extraction counts, but not the original
  publisher-specific data licenses or release versions.
- A current unauthenticated Hugging Face API query labels that mirror `MIT`.
  That tag covers the mirror repository metadata; it does not establish that
  every upstream raw EM asset may be relicensed or redistributed under MIT.
  Its README is gated to unauthenticated requests, so it cannot serve as a
  complete provenance record here.
- The official MitoEM challenge describes a mitochondria instance segmentation
  benchmark. Its labels therefore have a different object taxonomy from MEC
  myelin instances and must not be treated as direct supervised myelin labels.
- The MEC manifest itself records `cc-by-4.0` and is the only dataset in this
  tree with an explicit, task-level supervised-data license field.

## Decision

The inherited external collections remain legacy raw/self-supervised research
assets, not newly audited supervised training sources or production-provenance
assets. No external download or TOS upload was performed by this audit. Before
adding any external dataset to a new training or deployment lineage, record:

1. the official upstream release URL and citation;
2. the original license and any redistribution or commercial-use constraints;
3. a release/version identifier and checksum;
4. image modality, voxel spacing, label taxonomy, and whether labels can be
   used for myelin supervision; and
5. the isolated TOS prefix and an integrity manifest.

Until those fields exist, external data may not change the frozen MEC test
protocol and must not be represented as an independently licensed myelin
supervision source.
